"""Survey pretrained student candidates without weights, GPUs or AWS.

Only Hub metadata, README/license, config, safetensors index and tokenizer files
are fetched at pinned commits, anonymously. A gated file is recorded as gated and
never fetched another way. Weight shards are never downloaded: parameter counts
come from the Hub API, the index ``total_size`` and the safetensors JSON header
read with an HTTP Range request. Tokenizers are measured on Korean/English text
already on disk; no model is loaded or run.

    uv run --no-project --with 'tokenizers>=0.21,<0.24' --with 'pyarrow>=20,<30' \\
        python scripts/survey_student_candidates.py

The license classification below is a manual reading of the pinned files, not
legal advice. The script refuses to reuse it when the reviewed file hash changes.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import re
import string
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

HUB = "https://huggingface.co"
USER_AGENT = "bobcat-student-survey/1 (read-only, anonymous)"
SEED = "bobcat-student-survey-20260924-v1"
MAX_HEADER_BYTES = 16 * 1024 * 1024

# (repo, pinned commit, role). "variant" rows are size variants of a main family;
# identical tokenizer.json bytes reuse the main row's tokenizer measurement.
CANDIDATES = [
    ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "main"),
    ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "main"),
    ("kakaocorp/kanana-1.5-8b-instruct-2505", "c963a5f4f6496c749f94064a20b33028b0db9f19", "main"),
    ("kakaocorp/kanana-2-3b-instruct", "6a5d7889964c4c590299d16e309eabab1f73f8a9", "main"),
    ("skt/A.X-4.0-Light", "ba21c20ea1b31ded1ec3e2fb432335077dc4be98", "main"),
    ("skt/A.X-3.1-Light", "9b41bb2406472634d8812c0b8931fa40fa9a6c3a", "main"),
    ("trillionlabs/Tri-7B", "94e2cc881102428ab9be3df2366a09ac928c2c82", "main"),
    ("trillionlabs/Trida2.0-4B", "c7bc65ca7ca439a0acd78317ea4644139feb9836", "main"),
    ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
     "0728a47d632019a8da5f53b663db1c175dc04115", "main"),
    ("LGAI-EXAONE/EXAONE-4.0-1.2B", "3abf2810673c7c0778df64a73c2d52eab32d91c4", "main"),
    ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2", "main"),
    ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083", "main"),
    ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1", "main"),
    ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95", "main"),
    ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "variant"),
    ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "variant"),
    ("kakaocorp/kanana-1.5-2.1b-instruct-2505", "7df4bc35ccd610e451809d7106e1c3cf82bfd44c",
     "variant"),
    ("kakaocorp/kanana-2-1.3b-instruct", "bf4786aa2a1908adce942d53976270132732f720", "variant"),
    ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7", "variant"),
]

# Newer-release discovery. Results are an unpinned observation, kept for audit.
SEARCHES = [
    "author=Qwen&search=Qwen3.5", "author=Qwen&search=Qwen3.8", "author=kakaocorp",
    "author=skt", "author=trillionlabs", "author=naver-hyperclovax", "author=LGAI-EXAONE",
    "author=upstage", "author=google&search=gemma-4", "author=microsoft&search=phi",
    "author=HuggingFaceTB&search=SmolLM", "author=meta-llama&search=Llama-3.2",
]

WANTED = {
    "README.md", "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "chat_template.json", "added_tokens.json",
}
LICENSE_FILE = re.compile(r"(LICEN[CS]E|NOTICE|USE_POLICY)[^/]*", re.IGNORECASE)

# Local text: (path, sha256 from the original download receipt, nature, source).
KLUE = "klue/klue@349481ec73fff722f88e0453ca05c77a447d967c CC-BY-SA-4.0"
MASSIVE = "AmazonScience/massive@ed58ac423a2f4121720918bf5301577edce4ffd3 CC-BY-4.0"
SOURCES = {
    "klue_nli_train": ("artifacts/klue-source-v1/nli/train-00000-of-00001.parquet",
                       "35a8eb31818b94d9cfcb3477914360ce038b46b3bce4075eaef92309caa2c07c",
                       "native", KLUE),
    "klue_ynat_train": (
        "artifacts/klue-source-v1/ynat/train-00000-of-00001.parquet",
        "062c3b51c1ca34ed23c8fd19ffa5ea0ddcd914a95aeb9077d89576d3ef71d123",
        "native", KLUE,
    ),
    "nsmc_test": ("artifacts/decision-scale-downloads-v1/nsmc--ratings_test.txt",
                  "8ac9f64052f11dbf6ae0acb5e038f03d90a76f0eda7820cfb3a92d02edfcebda",
                  "native", "e9t/nsmc@cc0670e872d4ac27bfe36c87456783004b39ef6c CC0-1.0"),
    "massive_ko_test": ("artifacts/decision-massive-downloads-v1/ko-KR--test--0000.parquet",
                        "3e16361b8d1ecdf2de25684883f9b70a84b5397b965b393be957b9341c15bcfa",
                        "human-localized", MASSIVE),
    "massive_en_test": ("artifacts/decision-massive-downloads-v1/en-US--test--0000.parquet",
                        "c418da9a5f3a7425b5cf44941bcea032a31040dbda2da85e0dd9323785c7731e",
                        "original-english", MASSIVE),
}
NATIVE = ("klue_nli_train", "klue_ynat_train", "nsmc_test")
# The 255 identifier strings the GLM compiler actually verified (B0 readout).
GLM_IDENTIFIERS = "reports/2026-09-22-glm-readout-preflight.json"

# Fixed strings for mixed-script, number, code and normalization preservation.
PROBES = {
    "date_time": "2026년 9월 24일 오후 3시 15분에 회의가 있습니다.",
    "currency_percent": (
        "총액은 ₩1,234,567원이며 할인율은 12.5%입니다."
    ),
    "inline_code": "API 응답의 `confidence` 필드는 0.87이었다.",
    "python_code": "def 합계(값들):\n    return sum(값들)  # 합계 계산\n",
    "json_korean_keys": '{"질문": "환불 가능?", "후보": ["예", "아니오"]}',
    "ops_mixed": "Kubernetes pod가 CrashLoopBackOff 상태입니다. kubectl logs -n prod 로 확인.",
    "jamo_chat": "ㅋㅋㅋ 진짜 대박 ㅠㅠ",
    "hanja": "大韓民國 憲法 第1條",
    "nfd_hangul": unicodedata.normalize("NFD", "한국어 문장"),
    "whitespace": "탭\t구분\t값  두 칸   세 칸\n줄바꿈",
    "url": "https://example.com/경로?질의=값&page=2",
    "emoji": "좋아요 👍🏻 최고 🇰🇷",
}
NUMBERS = ["7", "42", "255", "1234567", "3.14159"]
THINK_WORDS = ("<think>", "</think>", "reasoning_content", "enable_thinking", "thinking")
TOOL_WORDS = ("<tool_call>", "tool_call", "<tool_response>", "tools", "function")

# Manual reading of the pinned license texts (2026-09-24). Quotes are verbatim and
# re-checked against the downloaded bytes on every run. Not legal advice.
APACHE_GRANT = ("royalty-free, irrevocable copyright license to reproduce, prepare "
                "Derivative Works of")
KANANA_OPEN = [
    "Offering or (re)selling to third parties access to Kanana Materials or any Derivative "
    "Works through API, cloud platforms, or other remote access services",
    "you may use Kanana Materials or any Derivative Works for the development and operation "
    "of your own services without obtaining a commercial license from KAKAO",
    "you must include **‘Kanana’** as a prefix to the name of such AI models",
    "clearly display the phrase **“Powered by Kanana”**",
]
GEMMA4_CARD = ('<b>License</b>: <a href="https://ai.google.dev/gemma/docs/gemma_4_license" '
               'target="_blank">Apache 2.0</a>')


def reviewed(status, files, quotes, commercial, redistribution, restrictions=(), note=""):
    return {"status": status, "files": files, "quotes": list(quotes),
            "commercial_use": commercial, "derivative_weight_redistribution": redistribution,
            "restrictions": list(restrictions), "note": note}


APACHE_TERMS = ["Apache-2.0 §4: include the license, mark modified files, keep NOTICE text"]
REVIEW: dict[str, dict] = {
    "Qwen/Qwen3-4B-Instruct-2507": reviewed("allowed", {
        "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        "README.md": "8e3dd0c3b5b11897cc71092ccfe517bb7a9783479baa3665aad73c8d1a2041cd"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS),
    "Qwen/Qwen3.5-4B": reviewed("allowed", {
        "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
        "README.md": "1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS),
    "kakaocorp/kanana-1.5-8b-instruct-2505": reviewed("allowed", {
        "LICENSE": "d1ee81266b96305cdec883e60fad804aa1d04b4698dccb5942ca51816b10df57",
        "README.md": "69cde2f5813308101f89ca22340d894df29a5a7687b9757e9c745f5ddfc13241"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS,
        "Markdown-formatted Apache-2.0; section titles 1-9 present."),
    "kakaocorp/kanana-2-3b-instruct": reviewed("needs_legal_review", {
        "LICENSE": "80745468c3213787d08ae8c3b1bc1ffb6b3563d6214ba53626547c39dfcdffd8",
        "README.md": "c5ec8f6d9a040405b44bfef8a524685352221d1b18fb050d04704c7c9171b998"},
        KANANA_OPEN, "conditional", "conditional", [
            "separate commercial license for third-party API/cloud, SI/on-premise, on-device",
            "derived model names must start with 'Kanana'; 'Powered by Kanana' display",
            "Responsible AI guideline must be passed on to users"]),
    "skt/A.X-4.0-Light": reviewed("allowed", {
        "LICENSE": "0d4b6e74abfb37024fba6f841f6b637d5570c3bf00d2dd2c3181dda4b313ae46",
        "README.md": "c7cad6f10664142e662168eded0af2664e74766ac447ca264a2519bccf9f0554"},
        [APACHE_GRANT, "(including modified model weights and tokenizer files) are distributed "
         "under the terms of the Apache License, Version 2.0",
         "Built with Qwen 2.5 — original model by Alibaba Cloud, licensed under the Apache "
         "License 2.0."], "yes", "yes",
        [*APACHE_TERMS, "keep Qwen 2.5 attribution NOTICE", "no SK Telecom trademark use"]),
    "skt/A.X-3.1-Light": reviewed("allowed", {
        "LICENSE": "9cf7ec77b0404b2a305a158cd614ef4ab24e3b71d85f3089ac295029ef90072d",
        "README.md": "7e37707d83a6cfc39f7a09b32f595a74259b9b192bb46add5a128d22a878849c"},
        [APACHE_GRANT, "(including modified model weights and tokenizer files) are distributed "
         "under the terms of the Apache License, Version 2.0"], "yes", "yes",
        [*APACHE_TERMS, "no SK Telecom trademark use"]),
    "trillionlabs/Tri-7B": reviewed("allowed", {
        "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "README.md": "0981d0d2db318324a7cff775a12afa011638a5753df0a56cf444dba602bf9a45"},
        [APACHE_GRANT, "This model is licensed under the Apache License 2.0."], "yes", "yes",
        APACHE_TERMS),
    "trillionlabs/Trida2.0-4B": reviewed("excluded", {
        "README.md": "6d49a44e84b01d51193b5a78e8f6f0082637a550df3f8e4b55bda09f3aad35bc"},
        ["license: other", "is a two-stream **block-diffusion** language model (research "
         "checkpoint)"], "unknown", "unknown",
        ["no license text in the repository; granted rights cannot be verified"],
        "Base model is Qwen/Qwen3.5-4B (Apache-2.0); this checkpoint's own terms are absent."),
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B": reviewed("needs_legal_review", {
        "LICENSE": "38ecac78adef21eceff609f917f3cf904b3381ee9a6c83f824b89f4f9d28bbb4",
        "README.md": "9e83852684d13326f59953311e48423be1611b2fc89c5bc62c795a028d273d67"},
        ["greater than 10 million monthly active users",
         "substantially similar to or directly competes with any product and service provided "
         "by NAVER",
         "you shall also include “HyperCLOVA X” at the beginning of any such AI model name",
         "NAVER reserves the right to modify or amend this Agreement at any time, in its sole "
         "discretion."], "conditional", "conditional", [
            ">10M MAU or competing product requires a separate license",
            "derived model names must start with 'HyperCLOVA X'; 'Powered by HyperCLOVA X'",
            "license is revocable and may be amended unilaterally"],
        "Hub gate 'auto': config/tokenizer returned 401 anonymously and were not measured."),
    "LGAI-EXAONE/EXAONE-4.0-1.2B": reviewed("excluded", {
        "LICENSE": "8c1762fb4bd94c8f17e114c5caa567e0948e1da7d02b79b55e99be9616e0f9e3",
        "README.md": "78f32c0424a98b3b734d8f002da71e4cbc92d96838fca7303b2cf599fc3b4f17"},
        ["EXAONE AI Model License Agreement 1.2 - NC",
         "The Licensee is expressly prohibited from using the Model, Derivatives, or Output for "
         "any commercial purposes"], "no", "research/education only", ["non-commercial"]),
    "google/gemma-4-E4B-it": reviewed("allowed", {
        "README.md": "b21e4f69614ccd77baa2f3797d05311040dee07b989cb9f0d25111aa4b605b2c",
        "license_link": "59d8f0ba87ad9a2f1a431123c8d16646e5b89ba53653e818f16d136d77263c99"},
        [GEMMA4_CARD, APACHE_GRANT], "yes", "yes", APACHE_TERMS,
        "No LICENSE file in the repository. The card links to an external page whose extracted "
        "terms equal the reference Apache-2.0 text; other pages on that site were not reviewed."),
    "microsoft/Phi-4-mini-instruct": reviewed("allowed", {
        "LICENSE": "fa8235e5b48faca34e3ca98cf4f694ef08bd216d28b58071a1f85b1d50cb814d",
        "NOTICE.md": "a3386cc7125ec75ed34d4f0a340c47782982095e251d322e01d8ecab22d87ba4",
        "README.md": "c4f34916182a3ee87d95d00c92847cfa687ab21dc338a577275f0c34cf0e1dfd"},
        ["to use, copy, modify, merge, publish, distribute, sublicense, and/or sell"],
        "yes", "yes", ["MIT: keep copyright and permission notice"]),
    "HuggingFaceTB/SmolLM3-3B": reviewed("allowed", {
        "README.md": "24f16f90b883af3618425a589783a944356a7fc2e3cfa69b579c9ff06963e178"},
        ["## License [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)"], "yes", "yes",
        APACHE_TERMS, "No LICENSE file in the repository; model-card declaration only."),
    "meta-llama/Llama-3.2-3B-Instruct": reviewed("needs_legal_review", {
        "LICENSE.txt": "0b4284c1f87029e67654c7953afa16279961632cf73dcfe33374c4c2f298fa35",
        "README.md": "8cea167f3aa4dbeea16d32caf6e446184e7e9aa9ebae7de6e3259ea049874570"},
        ["greater than 700 million monthly active users in the preceding calendar month",
         "prominently display “Built with Llama”",
         "you shall also include “Llama” at the beginning of any such AI model name."],
        "conditional", "conditional", [
            ">700M MAU requires a separate license",
            "derived model names must start with 'Llama'; 'Built with Llama' display",
            "Acceptable Use Policy (USE_POLICY.md was gated and not read)"],
        "Hub gate 'manual': config/tokenizer returned 401 anonymously and were not measured."),
    "Qwen/Qwen3-8B": reviewed("allowed", {
        "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        "README.md": "0f36caaff9c2516411a7738db384606263ba653c1e63e61d72f511606164d5a6"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS),
    "Qwen/Qwen3-1.7B": reviewed("allowed", {
        "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
        "README.md": "257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS),
    "kakaocorp/kanana-1.5-2.1b-instruct-2505": reviewed("allowed", {
        "LICENSE": "d1ee81266b96305cdec883e60fad804aa1d04b4698dccb5942ca51816b10df57",
        "README.md": "28c3a7d9cc3de59d7613d898537eeb2394393f5a4c0bb533e3a80e737cc0a24c"},
        [APACHE_GRANT], "yes", "yes", APACHE_TERMS,
        "Markdown-formatted Apache-2.0; section titles 1-9 present."),
    "kakaocorp/kanana-2-1.3b-instruct": reviewed("needs_legal_review", {
        "LICENSE": "80745468c3213787d08ae8c3b1bc1ffb6b3563d6214ba53626547c39dfcdffd8",
        "README.md": "fe33ea775030c376cbf27f5fde4e474c647467596de55cc2741d1ab2a179c2f6"},
        KANANA_OPEN, "conditional", "conditional", [
            "separate commercial license for third-party API/cloud, SI/on-premise, on-device",
            "derived model names must start with 'Kanana'; 'Powered by Kanana' display"]),
    "google/gemma-4-E2B-it": reviewed("allowed", {
        "README.md": "3e0608a6be80e3eb040a5aa6c76707809503deb85be6360184a118fb6156526c",
        "license_link": "59d8f0ba87ad9a2f1a431123c8d16646e5b89ba53653e818f16d136d77263c99"},
        [GEMMA4_CARD, APACHE_GRANT], "yes", "yes", APACHE_TERMS,
        "Same external license page as gemma-4-E4B-it."),
}


def now():
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def get(url: str, headers: dict | None = None, attempts: int = 3) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404) or attempt == attempts - 1:
                return error.code, b""
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def api(path: str):
    status, body = get(f"{HUB}/api/{path}")
    if status != 200:
        raise RuntimeError(f"Hub API {path} returned {status}")
    return json.loads(body)


def fetch(repo: str, revision: str, sibling: dict, folder: Path) -> dict:
    """Download one small file at the pinned commit and verify it against the Hub blob."""
    name = sibling["rfilename"]
    lfs = sibling.get("lfs")
    path = folder / name
    record = {"bytes": sibling.get("size")}

    def verified(data):
        return sha256(data) == lfs["sha256"] if lfs else git_blob(data) == sibling["blobId"]

    if path.exists() and verified(path.read_bytes()):
        data = path.read_bytes()
    else:
        status, data = get(f"{HUB}/{repo}/resolve/{revision}/{urllib.parse.quote(name)}")
        if status in (401, 403):
            return {**record, "status": "gated", "http_status": status}
        if status != 200:
            return {**record, "status": "missing", "http_status": status}
        if not verified(data):
            raise RuntimeError(f"{repo}/{name} does not match its Hub blob hash")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {**record, "status": "ok", "sha256": sha256(data),
            "verified_against": "lfs.sha256" if lfs else "git blob sha1"}


def safetensors_header(repo: str, revision: str, name: str) -> tuple[str, dict | None, int]:
    url = f"{HUB}/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"
    status, head = get(url, {"Range": "bytes=0-7"})
    if status in (401, 403):
        return "gated", None, 0
    if status != 206 or len(head) != 8:
        return f"http_{status}", None, len(head)
    (length,) = struct.unpack("<Q", head)
    if length > MAX_HEADER_BYTES:
        return "header_too_large", None, 8
    status, body = get(url, {"Range": f"bytes=8-{7 + length}"})
    if status != 206 or len(body) != length:
        return f"http_{status}", None, 8 + len(body)
    return "ok", json.loads(body), 8 + length


def component(name: str) -> str:
    lowered = name.lower()
    if re.search(r"(^|\.)mtp(\.|$)", lowered):
        return "mtp"
    if any(key in lowered for key in ("visual", "vision", "audio", "multi_modal_projector")):
        return "multimodal_encoder"
    if "embed_tokens_per_layer" in lowered:
        return "per_layer_embedding"
    if "embed_tokens" in lowered or lowered.endswith("wte.weight"):
        return "input_embedding"
    if "lm_head" in lowered:
        return "output_head"
    return "decoder"


def parameters(repo, revision, info, files, folder) -> dict:
    result = {"hub_api_safetensors": info.get("safetensors")}
    index = files.get("model.safetensors.index.json", {})
    if index.get("status") == "ok":
        metadata = json.loads((folder / "model.safetensors.index.json").read_text())
        result["index_total_size_bytes"] = metadata.get("metadata", {}).get("total_size")
    shards = sorted(item["rfilename"] for item in info["siblings"]
                    if item["rfilename"].endswith(".safetensors") and "/" not in item["rfilename"])
    counts, dtypes, read, statuses, scalars = Counter(), Counter(), 0, {}, 0
    for shard in shards:
        status, header, used = safetensors_header(repo, revision, shard)
        statuses[shard], read = status, read + used
        for key, value in (header or {}).items():
            if key == "__metadata__":
                continue
            numel = 1
            for size in value["shape"]:
                numel *= size
            scalars += not value["shape"]
            counts[component(key)] += numel
            dtypes[value["dtype"]] += numel
    result["safetensors_header"] = {
        "method": "HTTP Range read of the 8-byte length and JSON header only; no tensor data",
        "shards": statuses, "header_bytes_read": read,
        "complete": bool(shards) and all(status == "ok" for status in statuses.values()),
        "by_component": dict(counts), "by_dtype": dict(dtypes), "total": sum(counts.values()),
        # Shape [] tensors count as one element here; the Hub API total omits them.
        "scalar_tensors": scalars,
    }
    if result["safetensors_header"]["complete"]:
        result["text_decoder_total"] = sum(
            value for key, value in counts.items()
            if key not in ("multimodal_encoder", "mtp"))
        # Transformer blocks only: no input/per-layer embedding tables, no output head.
        result["decoder_blocks"] = counts["decoder"]
    return result


def text_config(config: dict) -> dict:
    return config.get("text_config") or config.get("llm_config") or config


def summarize_config(config: dict) -> dict:
    text = text_config(config)
    heads = text.get("num_attention_heads")
    kv = text.get("num_key_value_heads") or heads
    head_dim = text.get("head_dim") or (text["hidden_size"] // heads if heads else None)
    layers = text.get("num_hidden_layers")
    types = Counter(text.get("layer_types") or [])
    window = text.get("sliding_window")
    maximum = text.get("max_position_embeddings")
    sliding = types.get("sliding_attention", 0) or (
        window is not None and text.get("use_sliding_window") is not False
        and (maximum is None or window < maximum))
    linear = [key for key in types if "linear" in key or "mamba" in key or "recurrent" in key]
    moe = {key: text[key] for key in ("num_experts", "num_local_experts", "n_routed_experts")
           if isinstance(text.get(key), int) and text[key] > 1}
    if linear or any(key.startswith(("mamba", "ssm")) for key in text):
        attention = "hybrid_linear_recurrent"
    elif sliding:
        attention = "sliding_window_plus_global"
    elif kv == 1:
        attention = "MQA"
    else:
        attention = "GQA" if kv < heads else "MHA"
    full_layers = types.get("full_attention", 0) if types else layers
    full_head_dim = text.get("global_head_dim") or head_dim
    full_kv = text.get("num_global_key_value_heads") or kv
    special = {key: text[key] for key in (
        "num_kv_shared_layers", "hidden_size_per_layer_input", "final_logit_softcapping",
        "attn_output_gate", "mtp_num_hidden_layers", "full_attention_interval",
        "partial_rotary_factor", "no_rope_layer_interval") if text.get(key) not in (None, False)}
    if text.get("no_rope_layers"):
        special["no_rope_layers_count"] = len(text["no_rope_layers"]) - sum(text["no_rope_layers"])
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"), "text_model_type": text.get("model_type"),
        "layers": layers, "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "attention_heads": heads, "kv_heads": kv, "head_dim": head_dim,
        "attention_class": attention, "layer_types": dict(types),
        "sliding_window": window, "use_sliding_window": text.get("use_sliding_window"),
        "moe": moe or None, "vocab_size": text.get("vocab_size"),
        "max_position_embeddings": maximum,
        "tie_word_embeddings": text.get("tie_word_embeddings",
                                        config.get("tie_word_embeddings")),
        "multimodal_configs": sorted(key for key in config if key in (
            "vision_config", "audio_config", "speech_config")),
        "special": special,
        # BF16 K+V bytes per token for full-attention layers only. Sliding/linear
        # layers are bounded by window/state; KV sharing is not modelled here.
        "full_attention_kv_bytes_per_token_bf16": (
            full_layers * 2 * full_kv * full_head_dim * 2 if full_layers and full_kv else None),
    }


def front_matter(readme: str) -> dict:
    match = re.match(r"---\n(.*?)\n---", readme, re.DOTALL)
    fields = {}
    for line in (match.group(1).splitlines() if match else []):
        found = re.match(r"(license(?:_name|_link)?):\s*(.+)", line)
        if found:
            fields[found.group(1)] = found.group(2).strip().strip("'\"")
    return fields


def license_section(readme: str) -> str | None:
    lines, section, level = readme.splitlines(), None, 0
    for index, line in enumerate(lines):
        heading = re.match(r"(#+)\s+(.*)", line)
        if section is None and heading and re.search(r"licen[cs]e", heading.group(2), re.I):
            section, level = [line], len(heading.group(1))
            for rest in lines[index + 1:]:
                nested = re.match(r"(#+)\s", rest)
                if nested and len(nested.group(1)) <= level:
                    break
                section.append(rest)
            break
    return "\n".join(section).strip() if section else None


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def apache_terms(root: Path) -> str:
    """Sections 1-9 of the Apache-2.0 text already kept in this repository."""
    text = squash((root / "licenses/NVIDIA-AutoModel-Apache-2.0.txt").read_text())
    start = text.index("TERMS AND CONDITIONS FOR USE")
    return text[start:text.index("END OF TERMS AND CONDITIONS")]


def external_license(url: str, folder: Path) -> dict:
    """Fetch a card's non-Hub license page; hash its extracted terms, not volatile HTML."""
    status, body = get(url)
    if status != 200:
        return {"url": url, "status": "missing", "http_status": status}
    (folder / "license_link.html").write_bytes(body)
    page = re.sub(r"(?is)<(script|style)\b.*?</\1>", "", body.decode("utf-8", "replace"))
    text = squash(html.unescape(re.sub(r"<[^>]+>", "", page)))
    start, end = text.find("Apache License Version 2.0"), text.find("END OF TERMS AND CONDITIONS")
    terms = text[start:end + len("END OF TERMS AND CONDITIONS")] if 0 <= start < end else text
    (folder / "license_link.txt").write_text(terms + "\n")
    return {"url": url, "status": "ok", "html_sha256": sha256(body),
            "sha256": sha256(terms.encode()), "extracted": "Apache 2.0 terms span"
            if 0 <= start < end else "whole page text"}


def license_record(repo, info, files, folder, root) -> dict:
    readme = (folder / "README.md").read_text() if files.get("README.md", {}).get(
        "status") == "ok" else ""
    section = license_section(readme)
    texts = {name: (folder / name).read_text(errors="replace")
             for name, item in files.items() if item.get("status") == "ok"
             and (LICENSE_FILE.fullmatch(name) or name == "README.md")}
    link = (info.get("cardData") or {}).get("license_link") or ""
    has_file = any(LICENSE_FILE.fullmatch(name) for name in texts)
    if link.startswith("https://") and not link.startswith(HUB) and not has_file:
        files["license_link"] = external_license(link, folder)
        if files["license_link"]["status"] == "ok":
            texts["license_link"] = (folder / "license_link.txt").read_text()
    reference = apache_terms(root)
    record = {
        "hub_license_tag": [tag for tag in info.get("tags", []) if tag.startswith("license:")],
        "card_data": {key: (info.get("cardData") or {}).get(key)
                      for key in ("license", "license_name", "license_link")},
        "readme_front_matter": front_matter(readme),
        "readme_license_section": {"sha256": sha256(section.encode()), "chars": len(section),
                                   "text": section[:1200]} if section else None,
        "license_files": {name: files[name]["sha256"] for name in texts if name != "README.md"},
        "apache_2_terms_verbatim": {name: reference in squash(text)
                                    for name, text in texts.items() if name != "README.md"},
    }
    review = REVIEW.get(repo)
    if review is None:
        return {**record, "review": None, "status": "unreviewed"}
    stale = {name: expected for name, expected in review["files"].items()
             if files.get(name, {}).get("sha256") != expected}
    corpus = squash(" ".join(texts.values()))
    missing = [quote for quote in review["quotes"] if squash(quote) not in corpus]
    status = review["status"] if not stale and not missing else "review_stale"
    return {**record, "review": review, "stale_files": stale, "missing_quotes": missing,
            "status": status}


JAMO = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xD7B0, 0xD7FF))


def hangul(text: str) -> tuple[int, int]:
    syllables = sum(0xAC00 <= ord(char) <= 0xD7A3 for char in text)
    jamo = sum(any(low <= ord(char) <= high for low, high in JAMO) for char in text)
    return syllables, jamo


def rank(*parts: str) -> str:
    return sha256("\n".join((SEED, *parts)).encode())


def verify_source(root: Path, name: str) -> Path:
    relative, expected, _, _ = SOURCES[name]
    path = root / relative
    if sha256(path.read_bytes()) != expected:
        raise RuntimeError(f"{relative} does not match its download receipt")
    return path


def load_corpus(root: Path, size: int) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    pools = {}
    nli = pq.read_table(verify_source(root, "klue_nli_train"), columns=["premise", "hypothesis"])
    pools["klue_nli_train"] = nli.column("premise").to_pylist() + nli.column(
        "hypothesis").to_pylist()
    pools["klue_ynat_train"] = pq.read_table(
        verify_source(root, "klue_ynat_train"), columns=["title"]).column("title").to_pylist()
    lines = verify_source(root, "nsmc_test").read_text().splitlines()
    assert lines[0] == "id\tdocument\tlabel"
    # Only the document column is read; the rating label is never parsed.
    pools["nsmc_test"] = [line.split("\t")[1] for line in lines[1:]]
    samples, info = {}, {}
    for name, texts in pools.items():
        eligible = sorted({text.strip() for text in texts if text and text.strip()
                           and hangul(text)[0] > 0})
        chosen = sorted(eligible, key=rank)[:size]
        samples[name] = chosen
        info[name] = {"rows_read": len(texts), "eligible_unique": len(eligible),
                      "sample_size": len(chosen)}

    tables = {locale: pq.read_table(verify_source(root, f"massive_{locale}_test"),
                                    columns=["id", "utt"]).to_pylist()
              for locale in ("ko", "en")}
    english = {row["id"]: row["utt"].strip() for row in tables["en"]}
    pairs = sorted({(row["id"], row["utt"].strip(), english[row["id"]])
                    for row in tables["ko"] if row["id"] in english and row["utt"].strip()
                    and english[row["id"]] and hangul(row["utt"])[0] > 0},
                   key=lambda pair: rank(pair[1], pair[2]))
    chosen = pairs[:size]
    samples["massive_ko_test"] = [pair[1] for pair in chosen]
    samples["massive_en_test"] = [pair[2] for pair in chosen]
    for name in ("massive_ko_test", "massive_en_test"):
        info[name] = {"rows_read": len(tables[name.split("_")[1]]), "eligible_unique": len(pairs),
                      "sample_size": len(chosen)}
    info["massive_ko_test"]["pair_ids_sha256"] = sha256(
        json.dumps([pair[0] for pair in chosen]).encode())

    for name, texts in samples.items():
        relative, digest, nature, source = SOURCES[name]
        syllables, jamo = map(sum, zip(*(hangul(text) for text in texts), strict=True))
        info[name].update({
            "path": relative, "file_sha256": digest, "nature": nature, "source": source,
            "sample_sha256": sha256(json.dumps(texts, ensure_ascii=False).encode()),
            "hangul_syllables": syllables, "hangul_jamo": jamo,
            "non_space_chars": sum(len("".join(text.split())) for text in texts),
            "eojeol": sum(len(text.split()) for text in texts),
            "texts_changed_by_nfc": sum(unicodedata.normalize("NFC", text) != text
                                        for text in texts),
        })
    return samples, info


def first_difference(original: str, decoded: str) -> dict:
    index = next((i for i, (a, b) in enumerate(zip(original, decoded, strict=False)) if a != b),
                 min(len(original), len(decoded)))
    return {"index": index, "original": [f"U+{ord(c):04X}" for c in original[index:index + 3]],
            "decoded": [f"U+{ord(c):04X}" for c in decoded[index:index + 3]]}


def measure_texts(tokenizer, texts: list[str], unk: int | None) -> dict:
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    ids = [encoding.ids for encoding in encodings]
    decoded = tokenizer.decode_batch(ids, skip_special_tokens=False)
    failures = [(text, out) for text, out in zip(texts, decoded, strict=True) if text != out]
    lengths = [len(item) for item in ids]
    return {
        "tokens": sum(lengths), "lengths": lengths,
        "roundtrip_exact": len(texts) - len(failures),
        "roundtrip_exact_share": (len(texts) - len(failures)) / len(texts),
        "texts_with_unk": sum(unk in item for item in ids) if unk is not None else None,
        "roundtrip_failure_examples": [
            {"text_sha256": sha256(text.encode()), **first_difference(text, out)}
            for text, out in failures[:3]],
    }


def single_token(tokenizer, text: str, reserved: set[int]) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) != 1 or ids[0] in reserved:
        return None
    return ids[0] if tokenizer.decode(ids, skip_special_tokens=False) == text else None


def identifier_count(tokenizer, texts, reserved, limit=None) -> int:
    seen = set()
    for text in texts:
        token = single_token(tokenizer, text, reserved)
        if token is not None and token not in seen:
            seen.add(token)
            if limit and len(seen) == limit:
                break
    return len(seen)


def identifiers(tokenizer, reserved: set[int], glm: list[str]) -> dict:
    syllables = [chr(code) for code in range(0xAC00, 0xD7A4)]
    schemes = {
        "A-Z": list(string.ascii_uppercase), "a-z": list(string.ascii_lowercase),
        "0-9": list(string.digits), "10-255": [str(i) for i in range(10, 256)],
        "0-255": [str(i) for i in range(256)], "hangul_syllables": syllables,
    }
    counts = {}
    for name, texts in schemes.items():
        counts[name] = {"available": len(texts),
                        "no_space": identifier_count(tokenizer, texts, reserved),
                        "leading_space": identifier_count(
                            tokenizer, [" " + text for text in texts], reserved)}
    # The ordering used by the GLM compiler in src/bobcat/glm_readout.py.
    repo_order = [*string.ascii_uppercase, *string.ascii_lowercase, *map(str, range(1000))]
    ascii_hangul = [*string.ascii_uppercase, *string.ascii_lowercase, *string.digits, *syllables]
    combined = {"repo_glm_order_A-Z_a-z_0-999": identifier_count(
                    tokenizer, repo_order, reserved, 255),
                "A-Z_a-z_0-9_then_hangul": identifier_count(
                    tokenizer, ascii_hangul, reserved, 255)}
    # Same strings as GLM means the B0 compiled candidates can be reused unchanged.
    combined["glm_readout_identifier_strings"] = identifier_count(tokenizer, glm, reserved)
    reaching = [f"{name}:{variant}" for name, row in counts.items()
                for variant in ("no_space", "leading_space") if row[variant] >= 255]
    reaching += [name for name, value in combined.items() if value >= 255]
    return {"schemes": counts, "combined_first_255": combined,
            "schemes_reaching_255": reaching, "can_form_255": bool(reaching)}


def chat_template(folder: Path, files: dict) -> tuple[str | None, str | None]:
    if files.get("chat_template.jinja", {}).get("status") == "ok":
        return (folder / "chat_template.jinja").read_text(), "chat_template.jinja"
    for name in ("tokenizer_config.json", "chat_template.json"):
        if files.get(name, {}).get("status") == "ok":
            value = json.loads((folder / name).read_text()).get("chat_template")
            if isinstance(value, list):
                value = "\n".join(item["template"] for item in value)
            if value:
                return value, name
    return None, None


def audit_template(tokenizer, serialized: dict, template: str | None, source: str | None):
    added = serialized.get("added_tokens", [])
    if template is None:
        return {"source": None, "note": "no chat template in pinned files"}
    in_template = [token for token in added if token["content"] in template]
    rows = []
    for token in in_template:
        tokenizer.encode_special_tokens = False
        default = tokenizer.encode(token["content"], add_special_tokens=False).ids
        tokenizer.encode_special_tokens = True
        protected = tokenizer.encode(token["content"], add_special_tokens=False).ids
        tokenizer.encode_special_tokens = False
        lowered = token["content"].lower()
        kind = ("whitespace" if not token["content"].strip() else
                "think" if "think" in lowered else
                "tool" if "tool" in lowered or "function" in lowered else "role_or_control")
        rows.append({"content": token["content"], "id": token["id"], "special": token["special"],
                     "kind": kind, "matched_from_plain_text": default == [token["id"]],
                     "matched_with_encode_special_tokens": protected == [token["id"]]})
    non_special = [token["content"] for token in added if not token["special"]]
    return {
        "source": source, "sha256": sha256(template.encode()), "chars": len(template),
        "think_words": [word for word in THINK_WORDS if word in template],
        "tool_words": [word for word in TOOL_WORDS if word in template],
        "added_tokens_in_template": [row for row in rows if row["kind"] != "whitespace"],
        # special=False control tokens are matched in user text even when
        # encode_special_tokens=True; special ones match only under default settings.
        "non_special_control_in_template": [
            row["content"] for row in rows
            if row["matched_with_encode_special_tokens"] and row["kind"] != "whitespace"],
        "special_matched_from_plain_text_by_default": sum(
            row["special"] and row["matched_from_plain_text"] for row in rows),
        "whitespace_added_tokens_in_template": sum(row["kind"] == "whitespace" for row in rows),
        "non_special_added_tokens": {"count": len(non_special), "first_50": non_special[:50]},
    }


def measure_tokenizer(tokenizer, serialized: dict, samples: dict, glm: list[str]) -> dict:
    reserved = {token["id"] for token in serialized.get("added_tokens", [])}
    unk_name = (serialized.get("model") or {}).get("unk_token")
    unk = tokenizer.token_to_id(unk_name) if unk_name else None
    sources = {name: measure_texts(tokenizer, texts, unk) for name, texts in samples.items()}
    ko, en = sources["massive_ko_test"]["lengths"], sources["massive_en_test"]["lengths"]
    parallel = {
        "pairs": len(ko), "ko_tokens": sum(ko), "en_tokens": sum(en),
        "ko_over_en_total": sum(ko) / sum(en),
        "ko_over_en_median_pair": median(k / e for k, e in zip(ko, en, strict=True) if e),
    }
    probes = {}
    for name, text in PROBES.items():
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        probes[name] = {"tokens": len(ids),
                        "roundtrip_exact": tokenizer.decode(ids, skip_special_tokens=False)
                        == text}
    numbers = {text: len(tokenizer.encode(text, add_special_tokens=False).ids)
               for text in NUMBERS}
    for row in sources.values():
        row.pop("lengths")
    return {
        "loaded": True, "vocab_size_with_added": tokenizer.get_vocab_size(with_added_tokens=True),
        "model_type": (serialized.get("model") or {}).get("type"),
        # Same value = same base vocabulary and merge order, whatever the added tokens.
        "base_model_sha256": sha256(json.dumps(
            {key: (serialized.get("model") or {}).get(key) for key in ("vocab", "merges")},
            sort_keys=True, ensure_ascii=False).encode()),
        "normalizer": (serialized.get("normalizer") or {}).get("type"),
        "pre_tokenizer": (serialized.get("pre_tokenizer") or {}).get("type"),
        "byte_fallback": (serialized.get("model") or {}).get("byte_fallback"),
        "added_tokens": len(reserved), "unk_token": unk_name,
        "stored_padding": serialized.get("padding"),
        "stored_truncation": serialized.get("truncation"),
        "sources": sources, "parallel_massive": parallel, "probes": probes, "numbers": numbers,
        "identifiers": identifiers(tokenizer, reserved, glm),
    }


def rates(metrics: dict, info: dict) -> dict:
    out = {}
    for name, row in metrics["sources"].items():
        source = info[name]
        hangul_chars = source["hangul_syllables"] + source["hangul_jamo"]
        out[name] = {
            "tokens_per_text": row["tokens"] / source["sample_size"],
            "tokens_per_hangul_char": row["tokens"] / hangul_chars if hangul_chars else None,
            "tokens_per_non_space_char": row["tokens"] / source["non_space_chars"],
            "tokens_per_eojeol": row["tokens"] / source["eojeol"],
            "roundtrip_exact_share": row["roundtrip_exact_share"],
        }
    native = [out[name] for name in NATIVE]
    out["native_mean"] = {key: sum(row[key] for row in native) / len(native) for key in (
        "tokens_per_hangul_char", "tokens_per_eojeol", "roundtrip_exact_share")}
    return out


def suitability(row: dict) -> dict:
    config, params = row.get("config") or {}, row.get("parameters") or {}
    total = (params.get("hub_api_safetensors") or {}).get("total")
    text_total = params.get("text_decoder_total") or total
    license_status = row["license"]["status"]
    size_ok = bool(text_total) and 0.9e9 <= text_total <= 8.6e9
    dense = config.get("moe") is None if config else None
    attention = config.get("attention_class")
    ids = (row.get("tokenizer_metrics") or {}).get("identifiers") or {}
    blocked = [name for name, item in row["files"].items() if item["status"] == "gated"]
    return {
        "license_status": license_status, "text_params": text_total, "stored_params": total,
        "size_1_8b": size_ok, "dense": dense, "attention_class": attention,
        "standard_attention": attention in ("GQA", "MHA", "MQA") if attention else None,
        "can_form_255_single_token_identifiers": ids.get("can_form_255"),
        "gated_files": blocked,
        "excluded": license_status == "excluded",
    }


def survey(root: Path, folder: Path, sample_size: int) -> dict:
    import pyarrow
    import tokenizers
    from tokenizers import Tokenizer

    samples, corpus = load_corpus(root, sample_size)
    glm_file = root / GLM_IDENTIFIERS
    glm = json.loads(glm_file.read_text())["identifiers"]
    assert len(glm) == len(set(glm)) == 255
    searches = []
    for query in SEARCHES:
        found = api(f"models?{query}&sort=createdAt&direction=-1&limit=40")
        searches.append({"query": query, "observed_at": now(), "results": [
            {"id": item["id"], "createdAt": item.get("createdAt")} for item in found]})

    measured, rows = {}, []
    for repo, revision, role in CANDIDATES:
        local = folder / repo.replace("/", "--")
        local.mkdir(parents=True, exist_ok=True)
        info = api(f"models/{repo}/revision/{revision}?blobs=true")
        if info["sha"] != revision:
            raise RuntimeError(f"{repo} resolved to {info['sha']}, not {revision}")
        (local / "hub-api.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
        files = {item["rfilename"]: fetch(repo, revision, item, local)
                 for item in info["siblings"]
                 if item["rfilename"] in WANTED or LICENSE_FILE.fullmatch(item["rfilename"])}
        row = {
            "repo": repo, "revision": revision, "role": role,
            "local_dir": str(local.relative_to(root)) if local.is_relative_to(root) else str(local),
            "hub": {"sha": info["sha"], "lastModified": info.get("lastModified"),
                    "createdAt": info.get("createdAt"), "gated": info.get("gated"),
                    "pipeline_tag": info.get("pipeline_tag"),
                    "hub_api_sha256": sha256((local / "hub-api.json").read_bytes()),
                    "api_config_summary": {key: (info.get("config") or {}).get(key)
                                           for key in ("architectures", "model_type")}},
            "files": files,
            "license": license_record(repo, info, files, local, root),
        }
        if files.get("config.json", {}).get("status") == "ok":
            row["config"] = summarize_config(json.loads((local / "config.json").read_text()))
        row["parameters"] = parameters(repo, revision, info, files, local)
        tokenizer_file = files.get("tokenizer.json", {})
        if tokenizer_file.get("status") == "ok":
            digest = tokenizer_file["sha256"]
            row["tokenizer_sha256"] = digest
            try:
                tokenizer = Tokenizer.from_file(str(local / "tokenizer.json"))
                # Stored padding/truncation (Tri-7B pads batches on the left) would
                # change counts and round trips; measure raw encodings instead.
                tokenizer.no_padding()
                tokenizer.no_truncation()
            except Exception as error:  # record, do not hide, a tokenizer the library rejects
                row["tokenizer_metrics"] = {"loaded": False, "error": repr(error)}
            else:
                serialized = json.loads((local / "tokenizer.json").read_text())
                if digest not in measured:
                    metrics = measure_tokenizer(tokenizer, serialized, samples, glm)
                    metrics["rates"] = rates(metrics, corpus)
                    measured[digest] = {"first_repo": repo, **metrics}
                row["tokenizer_metrics"] = measured[digest]
                template, source = chat_template(local, files)
                row["chat_template"] = audit_template(tokenizer, serialized, template, source)
        row["suitability"] = suitability(row)
        rows.append(row)
        print(json.dumps({"repo": repo, "license": row["license"]["status"],
                          "tokenizer": bool(row.get("tokenizer_sha256"))}), file=sys.stderr)

    for row in rows:  # keep the JSON compact: one full copy per tokenizer hash
        if row.get("tokenizer_metrics", {}).get("loaded"):
            row["tokenizer_metrics"] = {
                "see_tokenizers": row["tokenizer_sha256"],
                "identifiers": row["tokenizer_metrics"]["identifiers"],
                "native_mean": row["tokenizer_metrics"]["rates"]["native_mean"],
                "ko_over_en_total": row["tokenizer_metrics"]["parallel_massive"][
                    "ko_over_en_total"]}
    script = Path(__file__).resolve()
    return {
        "schema": "bobcat-student-candidate-survey-v1",
        "generated_at": now(),
        "script": {"path": str(script.relative_to(root)), "sha256": sha256(script.read_bytes())},
        "environment": {"python": platform.python_version(), "tokenizers": tokenizers.__version__,
                        "pyarrow": pyarrow.__version__, "platform": platform.platform()},
        "constraints": {"weights_downloaded": False, "gpu_used": False, "aws_used": False,
                        "authenticated_hub_requests": False, "model_executed": False,
                        "jev_outputs_used": False},
        "method": {
            "sample": (f"Per source: stripped, deduplicated texts with >=1 Hangul syllable, "
                       f"ranked by sha256('{SEED}\\n' + text); first {sample_size}. MASSIVE: "
                       f"ko/en pairs joined by id, ranked by sha256(seed, ko, en). Labels "
                       f"are never read."),
            "tokenization": "tokenizers.Tokenizer.from_file(tokenizer.json), "
                            "add_special_tokens=False; decode(skip_special_tokens=False).",
            "hangul_chars": "Hangul syllables U+AC00-D7A3 plus Hangul jamo blocks.",
            "eojeol": "str.split() whitespace units.",
            "single_token_identifier": "encode(text) is one id, not an added token, decodes "
                                       "back to text, and is distinct within the scheme.",
            "glm_identifiers": {"path": GLM_IDENTIFIERS, "sha256": sha256(glm_file.read_bytes()),
                                "first": glm[:3], "last": glm[-3:], "count": len(glm)},
            "chat_template": "Added tokens whose content appears in the template; matched with "
                             "default settings and with encode_special_tokens=True.",
            "parameters": "Hub API safetensors field, index metadata.total_size, and "
                          "safetensors JSON headers via HTTP Range (tensor data never read).",
        },
        "search_snapshot": searches,
        "corpus": corpus,
        "tokenizers": measured,
        "candidates": rows,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", type=Path,
                        default=root / "artifacts/student-candidate-survey-20260924-v1")
    parser.add_argument("--out", type=Path,
                        default=root / "reports/2026-09-24-student-candidate-survey.json")
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    result = survey(root, args.artifacts, args.sample_size)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps([{**row["suitability"], "repo": row["repo"]} for row in result["candidates"]],
                     ensure_ascii=False, indent=1))
