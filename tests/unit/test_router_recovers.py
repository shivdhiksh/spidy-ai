import asyncio

from spidy.llm.client import LLMResponse
from spidy.llm.router import LLMRouter


class _P:
    provider_name = "p"
    _model = "m"

    def __init__(self):
        self.up = False

    async def complete(self, messages, **kw):
        return LLMResponse(text="ok", model="m") if self.up else LLMResponse.failure("getaddrinfo failed", model="m")


def test_router_recovers_after_the_internet_comes_back():
    p = _P()
    r = LLMRouter(providers=[p], failure_threshold=3)
    for _ in range(4):
        assert not asyncio.run(r.complete([])).success
    p.up = True                                   # Wi-Fi back: the very next request must work
    assert asyncio.run(r.complete([])).success
