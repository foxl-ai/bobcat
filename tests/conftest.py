import pytest
import torch

from bobcat.data import generate_dataset
from bobcat.model import ModelConfig
from bobcat.schema import read_examples
from bobcat.tokenization import ScratchTokenizer


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("corpus")
    generate_dataset(root / "data", train_worlds=24, eval_worlds=8, seed=7)
    train = read_examples(root / "data" / "train.jsonl")
    tokenizer = ScratchTokenizer.train(train, root / "tokenizer.json", vocab_size=512)
    return root, train, tokenizer


@pytest.fixture
def small_config(corpus):
    return ModelConfig(
        vocab_size=corpus[2].vocab_size,
        d_model=32,
        n_heads=4,
        encoder_layers=2,
        schema_layers=1,
        reader_blocks=1,
        joint_layers=3,
        max_context_tokens=512,
        max_schema_tokens=128,
        max_joint_tokens=768,
    )
