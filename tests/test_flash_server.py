"""The cached compiler of the Flash server yields exactly the stock compiler's token IDs."""

from bobcat import student_readout as sr
from bobcat.flash_server import CachedCompiler
from bobcat.protocol import parse_request
from tests.test_student_readout import student


def test_cached_compiler_matches_stock_compiler(tmp_path):
    pinned = student(tmp_path)
    stock = sr.StudentCompiler(tmp_path, pinned, ["A", "B", "C"], max_branch_tokens=4096,
                               piecewise=True)
    cached = CachedCompiler(tmp_path, pinned, ["A", "B", "C"], max_branch_tokens=4096,
                            piecewise=True, cache_size=2)
    payloads = [
        {"model": "m", "state": {"doc": "state yes no", "n": i % 3},
         "questions": {f"q{j}": {"type": "noul", "instructions": f"question {j}"}
                       for j in range(4)}}
        for i in range(6)]
    for payload in payloads:
        state, questions = parse_request(payload)
        for question in questions:
            assert cached.compile(state, question) == stock.compile(state, question)
    assert len(cached._cache) <= 2
    first = cached._data({"doc": "state"})
    first.append(999)                      # callers may mutate; the cache is unaffected
    assert cached._data({"doc": "state"}) == stock._data({"doc": "state"})
