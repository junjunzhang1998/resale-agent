"""Market data for pricing a secondhand item. No LLM in this file.

- ebay_search:   active eBay listings through the Browse API (item_summary/search)
- retail_search: current new prices through SerpApi's Google Shopping engine
- price_stats / comp_stats: plain-Python statistics, so the model never does the math

Both API calls go through a disk cache keyed on the request, so development runs don't burn
SerpApi's limited free searches. The cache stores the raw API response; records are compacted
from it on every read, so changing what a record holds never needs a new API call.
"""

import base64
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import httpx

EBAY_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_SCOPE = "https://api.ebay.com/oauth/api_scope"
SERPAPI_URL = "https://serpapi.com/search.json"
TIMEOUT = 20  # seconds per HTTP request

# Friendly condition names -> eBay condition IDs. Names and supported IDs vary by category:
# https://developer.ebay.com/api-docs/sell/static/metadata/condition-id-values.html
CONDITION_IDS = {
    "new": [1000],
    "open_box": [1500],  # "New other (see details)": opened or unboxed, never used
    "new_with_defects": [1750],
    "refurbished": [2000, 2010, 2020, 2030, 2500],
    "like_new": [2750, 2990],  # Like New, Pre-owned - Excellent
    "used": [2750, 2990, 3000, 3010, 4000, 5000, 6000],  # every pre-owned grade
    "for_parts": [7000],
}

# eBay access tokens last 2 hours. Get one per process and reuse it until shortly before expiry.
_token = {"value": None, "expires": 0.0}


# --- Disk cache ---


def default_cache_dir() -> Path:
    """./cache locally. On Cloud Run (which sets K_SERVICE) only /tmp is writable, so /tmp/cache:
    it lives in the instance's memory and is lost when the instance stops, which is fine for a cache."""
    return Path("/tmp/cache") if os.environ.get("K_SERVICE") else Path(__file__).parent / "cache"


def _cache_path(kind: str, params: dict) -> Path:
    cache_dir = Path(os.environ.get("CACHE_DIR") or default_cache_dir())
    key = hashlib.sha256(json.dumps([kind, params], sort_keys=True).encode()).hexdigest()[:24]
    return cache_dir / f"{kind}-{key}.json"


def _cached(kind: str, params: dict, fetch, use_cache: bool = True) -> dict:
    """Return fetch()'s response, from disk if a fresh copy exists. params must hold no secrets:
    they're the cache key, and they're saved next to the response to make the file readable."""
    path = _cache_path(kind, params)
    ttl = float(os.environ.get("CACHE_TTL_HOURS", "6")) * 3600
    # ttl > 0: on Windows a just-written file's mtime can be a hair after time.time()
    if use_cache and ttl > 0 and path.exists() and time.time() - path.stat().st_mtime < ttl:
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    response = fetch()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": kind, "params": params, "response": response}), encoding="utf-8")
    except OSError:
        pass  # a read-only disk only costs us the cache, not the answer
    return response


# --- eBay Browse API ---


def _ebay_token() -> str:
    if _token["value"] and time.time() < _token["expires"]:
        return _token["value"]
    client_id, secret = os.environ.get("EBAY_CLIENT_ID"), os.environ.get("EBAY_CLIENT_SECRET")
    if not client_id or not secret:
        raise RuntimeError("Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET (production keyset) in .env.")
    basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
    response = httpx.post(
        EBAY_TOKEN_URL,
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": EBAY_SCOPE},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    _token.update(value=body["access_token"], expires=time.time() + body.get("expires_in", 7200) - 60)
    return _token["value"]


def ebay_filter(condition=None, buying_options=None, min_price=None, max_price=None) -> str:
    """Build the Browse API filter string, e.g.
    'conditionIds:{3000|4000},buyingOptions:{FIXED_PRICE},price:[20..200],priceCurrency:USD,...'"""
    parts = []
    if condition:
        names = [condition] if isinstance(condition, str) else list(condition)
        unknown = [n for n in names if n not in CONDITION_IDS]
        if unknown:
            raise ValueError(f"Unknown condition {unknown}. Use one of: {', '.join(CONDITION_IDS)}")
        ids = sorted({i for n in names for i in CONDITION_IDS[n]})
        parts.append("conditionIds:{" + "|".join(map(str, ids)) + "}")
    if buying_options:
        options = [buying_options] if isinstance(buying_options, str) else list(buying_options)
        parts.append("buyingOptions:{" + "|".join(o.upper() for o in options) + "}")
    if min_price is not None or max_price is not None:
        low = "" if min_price is None else f"{float(min_price):g}"
        high = "" if max_price is None else f"{float(max_price):g}"
        # price needs priceCurrency, and eBay warns about priceCurrency without a price
        parts += [f"price:[{low}..{high}]", "priceCurrency:USD"]
    # US sellers shipping to the US, like our user's buyers
    parts += ["itemLocationCountry:US", "deliveryCountry:US"]
    return ",".join(parts)


def _money(amount: dict | None) -> float | None:
    try:
        return round(float(amount["value"]), 2)
    except (TypeError, KeyError, ValueError):
        return None


def ebay_record(item: dict) -> dict:
    """One itemSummary -> the compact record the agents see."""
    # Auction-only listings may have no price, only the current bid
    price = item.get("price") or item.get("currentBidPrice") or {}
    shipping_options = item.get("shippingOptions") or [{}]
    shipping = _money(shipping_options[0].get("shippingCost"))
    if shipping is None and shipping_options[0].get("shippingCostType") == "FREE":
        shipping = 0.0
    value = _money(price)
    return {
        "itemId": item.get("itemId"),
        "title": item.get("title"),
        "price": value,
        "currency": price.get("currency"),
        "shippingCost": shipping,  # None: calculated at checkout, unknown here
        "totalPrice": round(value + shipping, 2) if value is not None and shipping is not None else None,
        "condition": item.get("condition"),
        "conditionId": item.get("conditionId"),
        "buyingOptions": item.get("buyingOptions", []),
        "itemCreationDate": item.get("itemCreationDate"),
        # The tracking query string adds nothing: /itm/<id> opens the same listing
        "itemWebUrl": (item.get("itemWebUrl") or "").split("?")[0],
        # eBay's small (225px) listing photo, for the comps table; None if the listing has none
        "image": (item.get("image") or {}).get("imageUrl"),
    }


def ebay_search(
    query: str,
    condition: str | list[str] | None = None,
    limit: int = 50,
    buying_options: str | list[str] | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    sort: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Search active eBay US listings. Returns {query, filter, total, items: [records]}.

    condition: a key of CONDITION_IDS, or a list of them. buying_options: FIXED_PRICE, AUCTION,
    BEST_OFFER. sort: None (best match), 'price', '-price', 'newlyListed'.
    """
    params = {"q": query, "filter": ebay_filter(condition, buying_options, min_price, max_price), "limit": max(1, min(int(limit), 200))}
    if sort:
        params["sort"] = sort

    def fetch():
        response = httpx.get(
            EBAY_SEARCH_URL,
            params=params,
            headers={
                "Authorization": f"Bearer {_ebay_token()}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                # A buyer's location, so shippingCost is the real cost to a US buyer, not 'calculated'
                "X-EBAY-C-ENDUSERCTX": f"contextualLocation=country%3DUS%2Czip%3D{os.environ.get('EBAY_BUYER_ZIP', '10027')}",
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    body = _cached("ebay", params, fetch, use_cache)
    return {
        "query": query,
        "filter": params["filter"],
        "total": body.get("total", 0),
        "items": [ebay_record(item) for item in body.get("itemSummaries", [])],
    }


# --- SerpApi Google Shopping ---


# Google Shopping mixes resale listings in with retailers. They are not a retail price, so a
# record from one of these is flagged, and the retail ceiling leaves it out.
RESALE_SITES = ("ebay", "poshmark", "whatnot", "mercari", "offerup", "facebook", "swappa", "back market", "depop", "craigslist")


def retail_record(result: dict) -> dict:
    """One shopping_results item -> the compact record the agents see."""
    retailer = result.get("source") or ""
    return {
        "title": result.get("title"),
        "retailer": retailer,
        "is_resale": bool(result.get("second_hand_condition")) or any(s in retailer.lower() for s in RESALE_SITES),
        "price": result.get("price"),
        "extracted_price": result.get("extracted_price"),
        "old_price": result.get("old_price"),  # present when the item is on sale
        "extracted_old_price": result.get("extracted_old_price"),
        "second_hand_condition": result.get("second_hand_condition"),  # e.g. 'used', 'refurbished'
        # Google Shopping now often gives only a Google product page, not the retailer's URL
        "link": result.get("link") or result.get("product_link"),
        "product_id": result.get("product_id"),
    }


def retail_search(query: str, limit: int = 20, use_cache: bool = True) -> dict:
    """Current Google Shopping results for query in the US. Returns {query, items: [records]}."""
    params = {"engine": "google_shopping", "q": query, "gl": "us", "hl": "en"}

    def fetch():
        key = os.environ.get("SERPAPI_KEY")
        if not key:
            raise RuntimeError("Set SERPAPI_KEY in .env.")
        response = httpx.get(SERPAPI_URL, params={**params, "api_key": key}, timeout=TIMEOUT)
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error = body.get("error", "")
        if "hasn't returned any results" in error:
            return {"shopping_results": []}  # no results is an answer, not a failure
        if error or response.status_code != 200:
            raise RuntimeError(f"SerpApi: {error or response.status_code}")
        return body

    body = _cached("serpapi", params, fetch, use_cache)
    return {"query": query, "items": [retail_record(r) for r in body.get("shopping_results", [])[:limit]]}


# --- Statistics ---


def price_stats(prices: list[float]) -> dict:
    """n, min, max, median, quartiles, IQR, and the mean after dropping IQR outliers
    (outside Q1 - 1.5 IQR .. Q3 + 1.5 IQR). None values are skipped."""
    values = sorted(float(p) for p in prices if p is not None)
    if not values:
        return {"n": 0}
    if len(values) == 1:
        q1 = median = q3 = values[0]
    else:
        q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    kept = [v for v in values if low <= v <= high]
    r = lambda x: round(x, 2)  # noqa: E731
    return {
        "n": len(values),
        "min": r(values[0]),
        "max": r(values[-1]),
        "median": r(median),
        "q1": r(q1),
        "q3": r(q3),
        "iqr": r(iqr),
        "mean_without_outliers": r(statistics.fmean(kept)),
        "outliers_removed": len(values) - len(kept),
    }


def comp_stats(records: list[dict]) -> dict:
    """Stats for a set of eBay records: on the item price, and on price + shipping where the
    shipping cost is known."""
    known = [r for r in records if r.get("totalPrice") is not None]
    return {
        "item_price": price_stats([r.get("price") for r in records]),
        "price_with_shipping": price_stats([r["totalPrice"] for r in known]),
        "free_shipping": sum(1 for r in records if r.get("shippingCost") == 0),
        "shipping_unknown": len(records) - len(known),
    }


def _save_fixture(kind: str, params: dict, name: str) -> None:
    """Copy a cached raw response into tests/fixtures/, for the offline tests."""
    cached = json.loads(_cache_path(kind, params).read_text(encoding="utf-8"))
    out = Path(__file__).parent / "tests" / "fixtures" / name
    out.write_text(json.dumps(cached, indent=1), encoding="utf-8")
    print(f"  saved {out.relative_to(Path(__file__).parent)}")


if __name__ == "__main__":
    # Live key check, skipping the cache: uv run market.py ["AirPods Pro 2"] [--ebay-only] [--save-fixtures]
    # Costs one eBay call (free, high quota) and one SerpApi search (limited free quota).
    # --save-fixtures (with the default query) re-records the test fixtures from these responses.
    import sys

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = args[0] if args else "AirPods Pro 2"
    save = "--save-fixtures" in sys.argv

    print(f"eBay Browse API: '{query}', used, 10 listings")
    ebay = ebay_search(query, condition="used", limit=10, use_cache=False)
    print(f"  ok: {ebay['total']} matching listings; filter {ebay['filter']}")
    for item in ebay["items"][:3]:
        print(f"  ${item['price']} + ship {item['shippingCost']} | {item['condition']} | {item['title'][:60]} | {item['itemWebUrl']}")
    print("  stats:", json.dumps(comp_stats(ebay["items"])))
    if save:
        _save_fixture("ebay", {"q": query, "filter": ebay["filter"], "limit": 10}, "ebay_airpods_pro_2_used.json")

    if "--ebay-only" not in sys.argv:
        print(f"\nSerpApi Google Shopping: '{query}'")
        retail = retail_search(query, use_cache=False)
        print(f"  ok: {len(retail['items'])} results")
        for item in retail["items"][:3]:
            sale = f" (was {item['old_price']})" if item["old_price"] else ""
            print(f"  {item['price']}{sale} | {item['retailer']} | resale: {item['is_resale']} | {item['title'][:50]}")
        if save:
            _save_fixture("serpapi", {"engine": "google_shopping", "q": query, "gl": "us", "hl": "en"}, "serpapi_airpods_pro_2.json")
