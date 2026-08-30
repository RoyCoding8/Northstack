"""Provider-agnostic cost accounting.

Public because both the gateway and the worker need it -- the worker used to
import a private symbol from the gateway module.
"""

from __future__ import annotations

from northstack.adapters.providers.wire import Usage

_MILLION = 1_000_000
_CACHE_CREATION_RATE = 1.25
_CACHE_READ_RATE = 0.1


def compute_cost_usd(
    usage: Usage,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Compute USD cost for one call.

    Contract: ``input_tokens`` is non-cached regular input; cache creation and
    cache read are DISJOINT additive buckets billed at their own rates.
    Adapters normalize to this so cost is provider-agnostic.
    """
    buckets = (
        (usage.input_tokens, input_price_per_million),
        (usage.cache_creation_tokens, input_price_per_million * _CACHE_CREATION_RATE),
        (usage.cache_read_tokens, input_price_per_million * _CACHE_READ_RATE),
        (usage.output_tokens, output_price_per_million),
    )
    return round(sum(tokens / _MILLION * rate for tokens, rate in buckets), 10)
