"""Tests for market.py.

These never touch the network: HTTP calls are replaced by the recorded responses in
tests/fixtures/, so they give the same result every run. The live checks at the bottom are
marked `live` and skipped unless you run:  uv run pytest -m live
"""

import json
from pathlib import Path

import httpx
import pytest

import market

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["response"]


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """A fresh, empty cache per test, keys that look set, and no real HTTP."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EBAY_CLIENT_ID", "test-id")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("SERPAPI_KEY", "test-key")
    monkeypatch.setitem(market._token, "value", "test-token")
    monkeypatch.setitem(market._token, "expires", float("inf"))

    def no_network(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")

    monkeypatch.setattr(market.httpx, "get", no_network)
    monkeypatch.setattr(market.httpx, "post", no_network)


def serve(monkeypatch, body):
    """Answer every httpx.get with body, and record each call's params and headers."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"url": url, "params": params, "headers": headers})
        return FakeResponse(body)

    monkeypatch.setattr(market.httpx, "get", fake_get)
    return calls


# --- eBay ---


def test_ebay_filter_maps_conditions_to_ids():
    f = market.ebay_filter(condition="used", buying_options="fixed_price", min_price=20, max_price=200)
    assert "conditionIds:{2750|2990|3000|3010|4000|5000|6000}" in f
    assert "buyingOptions:{FIXED_PRICE}" in f
    assert "price:[20..200]" in f and "priceCurrency:USD" in f
    assert "itemLocationCountry:US" in f


def test_ebay_filter_accepts_several_conditions_and_open_price_bounds():
    f = market.ebay_filter(condition=["new", "open_box"], min_price=50)
    assert "conditionIds:{1000|1500}" in f
    assert "price:[50..]" in f


def test_ebay_filter_sends_price_currency_only_with_a_price():
    # eBay returns warning 12002 for priceCurrency without a price filter
    assert "priceCurrency" not in market.ebay_filter(condition="used")


def test_real_ebay_fixture_parses_into_records(monkeypatch):
    # A recorded live response (uv run market.py --ebay-only --save-fixtures)
    serve(monkeypatch, fixture("ebay_airpods_pro_2_used.json"))
    items = market.ebay_search("AirPods Pro 2", condition="used", limit=10)["items"]
    assert len(items) == 10
    assert all(i["price"] and i["itemWebUrl"].startswith("https://www.ebay.com/itm/") for i in items)
    assert all(i["conditionId"] == "3000" for i in items)
    assert sum(i["shippingCost"] is None for i in items) == 1  # one listing came back with no shippingOptions


def test_ebay_filter_rejects_unknown_condition():
    with pytest.raises(ValueError, match="Unknown condition"):
        market.ebay_filter(condition="gently loved")


def test_ebay_search_returns_compact_records(monkeypatch):
    calls = serve(monkeypatch, fixture("ebay_edge_cases.json"))
    result = market.ebay_search("AirPods Pro 2", condition="used", limit=10)

    assert calls[0]["params"]["q"] == "AirPods Pro 2"
    assert calls[0]["headers"]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_US"
    assert result["total"] == 2481
    first = result["items"][0]
    assert set(first) == {
        "itemId", "title", "price", "currency", "shippingCost", "totalPrice", "condition",
        "conditionId", "buyingOptions", "itemCreationDate", "seller", "itemWebUrl", "image",
    }
    assert first["price"] == 129.99 and first["shippingCost"] == 0.0 and first["totalPrice"] == 129.99
    assert "?" not in first["itemWebUrl"]  # tracking parameters stripped


def test_ebay_record_keeps_the_listing_image():
    record = market.ebay_record({"itemId": "1", "title": "t", "price": {"value": "10", "currency": "USD"},
                                 "image": {"imageUrl": "https://i.ebayimg.com/images/g/x/s-l225.jpg"}})
    assert record["image"].endswith("s-l225.jpg")
    assert market.ebay_record({"itemId": "2", "title": "t"})["image"] is None


def test_ebay_record_handles_calculated_shipping_and_auctions(monkeypatch):
    serve(monkeypatch, fixture("ebay_edge_cases.json"))
    items = {i["title"]: i for i in market.ebay_search("AirPods Pro 2", condition="used")["items"]}

    calculated = items["AirPods Pro 2nd gen used works great"]
    assert calculated["shippingCost"] is None and calculated["totalPrice"] is None

    auction = items["AirPods Pro 2 used - auction"]
    assert auction["price"] == 72.0  # the current bid, since there is no fixed price
    assert auction["buyingOptions"] == ["AUCTION"]
    assert auction["totalPrice"] == 76.99


def test_ebay_search_uses_the_disk_cache(monkeypatch):
    calls = serve(monkeypatch, fixture("ebay_edge_cases.json"))
    first = market.ebay_search("AirPods Pro 2", condition="used")
    second = market.ebay_search("AirPods Pro 2", condition="used")
    assert len(calls) == 1 and first == second

    market.ebay_search("AirPods Pro 2", condition="new")  # different params: a new call
    assert len(calls) == 2

    market.ebay_search("AirPods Pro 2", condition="used", use_cache=False)
    assert len(calls) == 3


def test_cache_expires_after_ttl(monkeypatch):
    calls = serve(monkeypatch, fixture("ebay_edge_cases.json"))
    monkeypatch.setenv("CACHE_TTL_HOURS", "0")
    market.ebay_search("AirPods Pro 2")
    market.ebay_search("AirPods Pro 2")
    assert len(calls) == 2


def test_missing_ebay_keys_give_a_clear_error(monkeypatch):
    monkeypatch.setitem(market._token, "value", None)
    monkeypatch.delenv("EBAY_CLIENT_ID")
    with pytest.raises(RuntimeError, match="EBAY_CLIENT_ID"):
        market.ebay_search("AirPods Pro 2")


# --- SerpApi ---


def test_retail_search_returns_compact_records(monkeypatch):
    calls = serve(monkeypatch, fixture("serpapi_airpods_pro_2.json"))
    result = market.retail_search("AirPods Pro 2")

    assert calls[0]["params"]["engine"] == "google_shopping" and calls[0]["params"]["api_key"] == "test-key"
    best_buy = result["items"][0]
    assert best_buy["retailer"] == "Best Buy" and best_buy["extracted_price"] == 249.95
    assert best_buy["is_resale"] is False
    assert best_buy["link"].startswith("https://")
    assert best_buy["product_id"]


def test_retail_search_flags_resale_listings_and_sales(monkeypatch):
    serve(monkeypatch, fixture("serpapi_airpods_pro_2.json"))
    items = market.retail_search("AirPods Pro 2")["items"]
    resale = {i["retailer"] for i in items if i["is_resale"]}
    assert {"eBay", "Poshmark", "Whatnot", "mercari.com", "Pawn America"} <= resale
    assert "Best Buy" not in resale
    assert any(i["extracted_old_price"] == 140 for i in items)  # an "Usually $140" sale


def test_retail_cache_key_has_no_api_key(monkeypatch, tmp_path):
    serve(monkeypatch, fixture("serpapi_airpods_pro_2.json"))
    market.retail_search("AirPods Pro 2")
    saved = next((tmp_path / "cache").glob("serpapi-*.json")).read_text()
    assert "test-key" not in saved


def test_serpapi_no_results_is_empty_not_an_error(monkeypatch):
    serve(monkeypatch, {"error": "Google hasn't returned any results for this query."})
    assert market.retail_search("zzzz nonexistent gadget")["items"] == []


def test_serpapi_error_raises(monkeypatch):
    serve(monkeypatch, {"error": "Invalid API key."})
    with pytest.raises(RuntimeError, match="Invalid API key"):
        market.retail_search("AirPods Pro 2")


# --- Statistics ---


def test_price_stats_basic():
    s = market.price_stats([100, 110, 120, 130, 140])
    assert s["n"] == 5 and s["min"] == 100 and s["max"] == 140 and s["median"] == 120
    assert s["q1"] == 110 and s["q3"] == 130 and s["iqr"] == 20
    assert s["mean_without_outliers"] == 120 and s["outliers_removed"] == 0


def test_price_stats_drops_iqr_outliers_from_the_mean():
    s = market.price_stats([100, 105, 110, 115, 120, 900])
    assert s["outliers_removed"] == 1
    assert s["mean_without_outliers"] == 110
    assert s["max"] == 900  # min and max still report the raw data


def test_price_stats_small_and_empty_inputs():
    assert market.price_stats([]) == {"n": 0}
    assert market.price_stats([None]) == {"n": 0}
    one = market.price_stats([50])
    assert one["median"] == 50 and one["iqr"] == 0 and one["mean_without_outliers"] == 50


def test_comp_stats_on_fixture(monkeypatch):
    serve(monkeypatch, fixture("ebay_edge_cases.json"))
    s = market.comp_stats(market.ebay_search("AirPods Pro 2", condition="used")["items"])
    assert s["item_price"]["n"] == 9
    assert s["price_with_shipping"]["n"] == 8  # one listing has calculated shipping
    assert s["shipping_unknown"] == 1 and s["free_shipping"] == 4
    assert s["item_price"]["outliers_removed"] >= 1  # the lot of 3 at $389


# --- Live (uv run pytest -m live) ---


@pytest.fixture
def live(monkeypatch, tmp_path):
    """Undo the offline fixture: real keys from .env and real HTTP."""
    from dotenv import load_dotenv

    monkeypatch.undo()
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)


@pytest.mark.live
def test_live_ebay_search(live):
    result = market.ebay_search("AirPods Pro 2", condition="used", limit=5)
    assert result["items"], result
    assert all(i["itemWebUrl"].startswith("https://www.ebay.com/itm/") for i in result["items"])


@pytest.mark.live
def test_live_retail_search(live):
    result = market.retail_search("AirPods Pro 2")
    assert result["items"] and any(i["extracted_price"] for i in result["items"])


def test_cache_defaults_to_tmp_on_cloud_run(monkeypatch):
    monkeypatch.delenv("CACHE_DIR")
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert market.default_cache_dir() == Path(market.__file__).parent / "cache"
    monkeypatch.setenv("K_SERVICE", "resale-agent")  # Cloud Run sets K_SERVICE
    assert market.default_cache_dir() == Path("/tmp/cache")
    assert market._cache_path("ebay", {"q": "x"}).parent == Path("/tmp/cache")
