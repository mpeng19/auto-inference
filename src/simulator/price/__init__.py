"""Turning measured GPU-seconds into a price, and a price into a market share.

Pure arithmetic; nothing here touches a GPU or the network.

    direct.usable(level, n_gpu) -> (ok, why)
        refuse to price a level whose counters cannot support it
    direct.price_direct(gpu_s_in, gpu_s_out, in_tok, out_tok, cached_tok, ...)
        -> DirectPrice: $/M effective input and $/M output at our own hit rate
    direct.gpu_seconds_per_request(...)
        forward GPU-seconds for one market-sized request
    market.Market.load() -> Market
        the demand side: requests/day, tokens per request, the provider board;
        `bill_per_1k` and `leaderboard` score a price against it
    market.Economics(gpu_s_per_request, n_gpu, rate, utilisation)
        capacity per node, share per node, price as a function of share

Reads: the OpenRouter snapshot `simulator/data/market-qwen-qwen3.8-27b.json`
(or `SIMULATOR_MARKET_DATA`); `market_pull` refreshes it from openrouter.ai
and writes `market-<slug>.json` into the working directory. Writes nothing
else.
"""
