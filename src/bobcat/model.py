from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    architecture: str = "shared"
    vocab_size: int = 2048
    d_model: int = 128
    n_heads: int = 4
    encoder_layers: int = 3
    schema_layers: int = 1
    reader_blocks: int = 1
    reader_passes: int = 1
    candidate_pooling: str = "marker"
    joint_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.0
    max_context_tokens: int = 256
    max_schema_tokens: int = 96
    max_joint_tokens: int = 512
    activation_checkpointing: bool = False
    mlm_transform: bool = False

    def __post_init__(self):
        if self.architecture not in {"shared", "joint"}:
            raise ValueError("Architecture must be shared or joint.")
        if self.candidate_pooling not in {"marker", "candidate_mean"}:
            raise ValueError("Candidate pooling must be marker or candidate_mean.")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("Width must be divisible by heads and by two.")
        if not 1 <= self.schema_layers <= self.encoder_layers:
            raise ValueError("Schema layers must share a nonempty prefix of the encoder.")
        if min(self.encoder_layers, self.reader_blocks, self.reader_passes, self.joint_layers) < 1:
            raise ValueError("Layer/pass counts must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.dropout = config.dropout

    def split(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        return self.split(self.k(memory)), self.split(self.v(memory))

    def forward(
        self,
        query: Tensor,
        keep: Tensor,
        *,
        memory: Tensor | None = None,
        kv: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if kv is None:
            kv = self.project_kv(query if memory is None else memory)
        query_heads = self.split(self.q(query))
        # SDPA uses True = allowed, unlike MultiheadAttention's key_padding_mask.
        output = F.scaled_dot_product_attention(
            query_heads,
            kv[0],
            kv[1],
            attn_mask=keep[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(query.shape)
        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.ff_mult),
            nn.GELU(),
            nn.Linear(config.d_model * config.ff_mult, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attention = Attention(config)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, keep: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.norm_attn(x), keep))
        return x + self.ff(self.norm_ff(x))


class ReaderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_cross = nn.LayerNorm(config.d_model)
        self.norm_memory = nn.LayerNorm(config.d_model)
        self.cross = Attention(config)
        self.norm_set = nn.LayerNorm(config.d_model)
        self.set_attention = Attention(config)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def prepare_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        return self.cross.project_kv(self.norm_memory(memory))

    def forward(
        self,
        query: Tensor,
        memory_keep: Tensor,
        candidate_keep: Tensor,
        kv: tuple[Tensor, Tensor],
    ) -> Tensor:
        query = query + self.dropout(self.cross(self.norm_cross(query), memory_keep, kv=kv))
        # Deliberately no candidate-index position encoding.
        query = query + self.dropout(self.set_attention(self.norm_set(query), candidate_keep))
        return query + self.ff(self.norm_ff(query))


@dataclass
class StateMemory:
    hidden: Tensor
    keep: Tensor
    projected: list[tuple[Tensor, Tensor]]


class DecisionModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
        layers = config.encoder_layers if config.architecture == "shared" else config.joint_layers
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(layers)])
        self.encoder_norm = nn.LayerNorm(d)
        self.reader = nn.ModuleList(
            [ReaderBlock(config) for _ in range(config.reader_blocks)]
            if config.architecture == "shared"
            else []
        )
        self.decision_norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.mlm_transform = (
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
            if config.mlm_transform else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._initialize)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed(self, ids: Tensor) -> Tensor:
        width = self.config.d_model
        positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, width, 2, device=ids.device, dtype=torch.float32)
            * (-math.log(10000.0) / width)
        )
        angles = positions[:, None] * frequencies[None, :]
        position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        hidden = self.embedding(ids)
        return self.dropout(hidden * math.sqrt(width) + position.to(hidden.dtype)[None])

    def encode(self, ids: Tensor, layers: int | None = None) -> Tensor:
        hidden = self.embed(ids)
        keep = ids.ne(0)
        for block in self.encoder[:layers]:
            if self.config.activation_checkpointing and self.training:
                hidden = checkpoint(block, hidden, keep, use_reentrant=False)
            else:
                hidden = block(hidden, keep)
        return self.encoder_norm(hidden)

    def encode_state(self, ids: Tensor) -> StateMemory:
        if self.config.architecture != "shared":
            raise ValueError("The joint baseline does not offer shared state memory.")
        hidden = self.encode(ids)
        return StateMemory(
            hidden=hidden,
            keep=ids.ne(0),
            projected=[block.prepare_memory(hidden) for block in self.reader],
        )

    def decide(
        self,
        memory: StateMemory,
        context_index: Tensor,
        schema_ids: Tensor,
        candidate_keep: Tensor,
        schema_candidate_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, candidates, length = schema_ids.shape
        hidden = self.encode(
            schema_ids.reshape(batch_size * candidates, length),
            layers=self.config.schema_layers,
        )
        if self.config.candidate_pooling == "candidate_mean":
            if schema_candidate_mask is None:
                raise ValueError("Candidate pooling needs an explicit candidate token mask.")
            weights = schema_candidate_mask.reshape(batch_size * candidates, length).to(
                hidden.dtype
            )
            pooled = (hidden * weights[:, :, None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]
            query = pooled.reshape(batch_size, candidates, -1)
        else:
            query = hidden[:, 0].reshape(batch_size, candidates, -1)
        # Project context K/V once per block, sharing both across questions and passes.
        kvs = [
            (key.index_select(0, context_index), value.index_select(0, context_index))
            for key, value in memory.projected
        ]
        keep = memory.keep.index_select(0, context_index)
        for _ in range(self.config.reader_passes):
            for block, kv in zip(self.reader, kvs, strict=True):
                query = block(query, keep, candidate_keep, kv)
        logits = self.score(self.decision_norm(query)).squeeze(-1)
        return logits.masked_fill(~candidate_keep, float("-inf"))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        if "mlm_ids" in batch:
            return self.mlm_logits(batch["mlm_ids"], batch["mlm_positions"])
        if self.config.architecture == "shared":
            memory = self.encode_state(batch["context_ids"])
            return self.decide(
                memory,
                batch["context_index"],
                batch["schema_ids"],
                batch["candidate_keep"],
                batch["schema_candidate_mask"],
            )
        hidden = self.encode(batch["joint_ids"])
        if self.config.candidate_pooling == "candidate_mean":
            mask = batch["joint_candidate_mask"].to(hidden.dtype)
            gathered = torch.bmm(mask, hidden) / mask.sum(-1, keepdim=True).clamp_min(1)
        else:
            positions = batch["candidate_positions"]
            gathered = hidden.gather(1, positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
        logits = self.score(self.decision_norm(gathered)).squeeze(-1)
        return logits.masked_fill(~batch["candidate_keep"], float("-inf"))

    def mlm_logits(self, ids: Tensor, masked_positions: Tensor) -> Tensor:
        hidden = self.encode(ids)
        selected = self.mlm_transform(hidden[masked_positions])
        return F.linear(selected, self.embedding.weight, self.mlm_bias)


def parameter_count(config: ModelConfig) -> int:
    # Count even the 450M candidate without allocating its weights.
    with torch.device("meta"):
        model = DecisionModel(config)
    return sum(parameter.numel() for parameter in model.parameters())
