"""Tests for the code around the agents in research.py: the title filter, review and stats,
gaps, the round cap, confidence, and the citation check. No model is called: market searches
come from tests/fixtures, and agent runs are replaced where a test needs one."""

import asyncio
import json
from pathlib import Path

import pytest

import market
import research
from research import Brief, Exclusion, Item, Memo, ResearcherState, Run, Trace

FIXTURES = Path(__file__).parent / "fixtures"
ITEM = Item(product="AirPods Pro 2", variant="", condition="used", condition_detail="good", included="charging case", search_query="AirPods Pro 2")


def fixture_records(name, record):
    body = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["response"]
    key = "itemSummaries" if "itemSummaries" in body else "shopping_results"
    return [record(r) for r in body[key]]


@pytest.fixture
def offline_market(monkeypatch):
    """market searches answer from the fixtures; each eBay call returns both fixtures' listings."""
    ebay = fixture_records("ebay_airpods_pro_2_used.json", market.ebay_record) + fixture_records("ebay_edge_cases.json", market.ebay_record)
    retail = fixture_records("serpapi_airpods_pro_2.json", market.retail_record)
    monkeypatch.setattr(market, "ebay_search", lambda query, condition=None, limit=50, **kw: {"query": query, "filter": "", "total": len(ebay), "items": ebay})
    monkeypatch.setattr(market, "retail_search", lambda query, limit=20, **kw: {"query": query, "items": retail})


def brief(researcher="exact_comps", conditions=("used",), broadened=False, successor=False):
    return Brief(researcher=researcher, query="AirPods Pro 2", conditions=list(conditions), broadened=broadened, successor=successor, instructions="")


def new_run():
    return Run(item=ITEM, assumptions=[], trace=Trace())


# --- Title filter ---


@pytest.mark.parametrize("title, reason", [
    ("AirPods Pro 2 Charging Case Only USB-C", "case only"),
    ("Apple AirPods Pro 2nd Gen LEFT Earbud Genuine", "single earbud / one side"),
    ("AirPods Pro 2 Right Side Only A2698", "single earbud / one side"),
    ("Single AirPods Pro 2 earbud", "single earbud / one side"),
    ("AirPods Pro 2 Replacement Case A2700", "case only"),
    ("AirPods Pro 2 replacement ear tips", "replacement part"),
    ("Lot of 3 AirPods Pro 2", "lot / multiple units"),
    ("AirPods Pro 2 EMPTY BOX with manual", "box only"),
    ("AirPods Pro 2 for parts not working", "for parts / not working"),
    ("Case cover for AirPods Pro 2", "accessory, not the item"),
    ("Working Apple AirPods Pro 2nd Generation + OG box read desc", "sold as-is / see description"),
    ("Apple AirPods Pro 2nd Generation Lightly Used Tested Sold As Is", "sold as-is / see description"),
])
def test_title_filter_catches_partial_items(title, reason):
    assert research.title_exclusion(title) == reason


@pytest.mark.parametrize("title", [
    "Apple AirPods Pro 2 White With USB-C Charging Case In Ear Headphones",  # the full item
    "AirPods Pro 2nd Gen with left and right earbuds and MagSafe case",
    "Apple AirPods Pro (2nd Generation) Lightning - Used",
])
def test_title_filter_keeps_complete_items(title):
    assert research.title_exclusion(title) is None


def test_for_parts_tier_keeps_parts_listings():
    assert research.title_exclusion("AirPods Pro 2 for parts not working", tier="for_parts") is None
    assert research.title_exclusion("AirPods Pro 2 case only for parts", tier="for_parts") == "case only"


def test_pattern_matching_the_sellers_own_item_is_skipped():
    assert research.title_exclusion("AirPods Pro 2 charging case only", item_text="AirPods Pro 2 charging case only") is None


# --- Search, review, finding ---


def test_search_drops_titles_and_auctions_in_code(offline_market):
    state = ResearcherState(new_run(), brief(), 1)
    result = asyncio.run(research.do_search_ebay(state, "AirPods Pro 2", ["used"]))
    reasons = {d["title"]: d["reason"] for d in state.dropped}
    assert reasons["AirPods Pro 2 LOT OF 3 used"] == "lot / multiple units"
    assert reasons["AirPods Pro 2 used - auction"].startswith("auction")
    shown_lines = result["listings_to_review"]["used"]
    assert len(shown_lines) == len(state.shown) and all(line.startswith("E") for line in shown_lines)


def test_review_keeps_what_isnt_excluded_and_counts_before_and_after(offline_market):
    state = ResearcherState(new_run(), brief(), 1)
    asyncio.run(research.do_search_ebay(state, "AirPods Pro 2", ["used"]))
    cheapest = min(state.shown, key=lambda s: state.run.sources[s]["price"])
    out = research.do_review(state, [Exclusion(id=cheapest, reason="accessory: likely case only"), Exclusion(id="E999", reason="x")])

    f = state.finding
    assert f["n_found"] == len(state.shown) + len(state.dropped)
    assert f["n_after_title_filter"] == len(state.shown)
    assert f["n_kept"] == len(state.shown) - 1 == out["n_kept"]
    assert cheapest not in f["kept_ids"]
    assert out["unknown_ids_ignored"] == ["E999"]
    assert {e["by"] for e in f["excluded"]} == {"code", "review"}
    assert f["excluded_summary"]["accessory: likely case only"] == 1
    # stats come from market.comp_stats on the kept set only
    assert f["stats"] == market.comp_stats([state.run.sources[s] for s in f["kept_ids"]])


def test_a_second_review_does_not_bring_back_an_earlier_exclusion(offline_market):
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2"))
    first = next(iter(state.shown))
    research.do_review(state, [Exclusion(id=first, reason="older generation")])
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2 USB-C"))
    research.do_review(state, [])
    assert first not in state.finding["kept_ids"]
    assert state.finding["excluded_summary"]["older generation"] == 1


def test_same_listing_keeps_its_citation_id_across_searches(offline_market):
    run = new_run()
    a, b = ResearcherState(run, brief(), 1), ResearcherState(run, brief("condition_tiers", ["new", "used"]), 1)
    asyncio.run(research.do_search_ebay(a, "AirPods Pro 2", ["used"]))
    asyncio.run(research.do_search_ebay(b, "AirPods Pro 2", ["new", "used"]))
    assert set(a.shown) == set(b.shown)  # the same listings got the same ids
    assert len([k for k in run.sources if k.startswith("E")]) == 19


def test_retail_search_drops_resale_listings(offline_market):
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2"))
    research.do_review(state, [])
    retailers = {state.run.sources[s]["retailer"] for s in state.finding["kept_ids"]}
    assert "Best Buy" in retailers and "eBay" not in retailers and "Poshmark" not in retailers
    assert state.finding["stats"]["n"] == state.finding["n_kept"]


def test_retail_offers_sharing_a_product_get_their_own_ids(monkeypatch):
    offer = {"title": "Apple AirPods Pro with MagSafe Case", "product_id": "111", "price": "$249.95", "extracted_price": 249.95, "old_price": None,
             "extracted_old_price": None, "second_hand_condition": None, "is_resale": False}
    offers = [{**offer, "retailer": "Best Buy", "link": "https://g/1"}, {**offer, "retailer": "Walmart", "price": "$199.00", "extracted_price": 199.0, "link": "https://g/2"}]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": offers})
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro"))
    assert [state.run.sources[s]["retailer"] for s in state.shown] == ["Best Buy", "Walmart"]


def test_researcher_that_fails_is_finished_in_code(offline_market, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(research.Runner, "run", broken)
    run = new_run()
    finding = asyncio.run(research.run_researcher(run, brief(), 1))
    assert finding["n_kept"] > 0 and "stopped early" in finding["note"]
    assert [c["name"] for c in run.trace.calls] == ["search_ebay (code fallback)", "review_listings (code fallback: kept all)"]
    assert all(c["agent"] == "researcher:exact_comps" for c in run.trace.calls)
    assert all(c["type"] == "step" for c in run.trace.calls)  # code ran these, not the agent


def test_trace_marks_agent_tool_calls_as_tools():
    class Ctx:
        tool_arguments, tool_call_id = '{"query": "x"}', "call_1"

    class Named:
        name = "search_ebay"

    agent = Named()
    agent.name = "researcher:exact_comps"
    trace = Trace()
    asyncio.run(trace.on_tool_start(Ctx(), agent, Named()))
    asyncio.run(trace.on_tool_end(Ctx(), agent, Named(), "ok"))
    trace.record("writer", "validate_citations", {}, {})
    assert [(c["type"], c["name"], c["result"]) for c in trace.calls] == [("tool", "search_ebay", "ok"), ("step", "validate_citations", "{}")]
    assert set(trace.calls[0]) == {"type", "agent", "name", "args", "result"}


# --- Lead: fan-out, gaps, round cap ---


def fake_finding(researcher="exact_comps", n_kept=10, broadened=False, kept_prefix="E", successor=False, only_resale=False):
    return {"successor": successor, "only_resale_listings": only_resale, "researcher": researcher, "round": 1, "queries": ["q"], "broadened": broadened, "n_found": n_kept, "n_after_title_filter": n_kept,
            "n_kept": n_kept, "excluded_summary": {}, "excluded": [], "kept_ids": [f"{kept_prefix}{i}" for i in range(1, n_kept + 1)],
            "errors": {}, "conditions": ["used"], "stats": {"price_with_shipping": {"median": 1}, "median": 1, "min": 1}}


def test_only_resale_retail_results_are_final_not_a_gap(monkeypatch):
    resale = [{"title": "AirPods Pro 2", "retailer": "eBay", "is_resale": True, "link": f"https://e/{i}", "product_id": str(i),
               "price": "$90", "extracted_price": 90.0, "old_price": None, "extracted_old_price": None, "second_hand_condition": "used"} for i in range(3)]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": resale})
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2"))
    research.do_review(state, [])
    assert state.finding["only_resale_listings"] is True and state.finding["n_kept"] == 0
    assert research.find_gaps([fake_finding(), state.finding]) == []
    assert "Don't retry retail" in research.lead_summary(state.finding)["result"]
    state.run.findings.append(state.finding)
    assert research.no_new_retail(state.run)


def test_gaps_flag_few_comps_and_no_retail():
    gaps = research.find_gaps([fake_finding(n_kept=3), fake_finding("retail", n_kept=0)])
    assert any("Only 3 exact comps" in g for g in gaps)
    assert any("No new retail price" in g for g in gaps)
    assert research.find_gaps([fake_finding(n_kept=8), fake_finding("retail", n_kept=2, kept_prefix="R")]) == []


def test_dispatch_runs_briefs_in_parallel_and_caps_rounds(monkeypatch):
    running, peak = 0, 0

    async def fake_researcher(run, b, round_no):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)
        running -= 1
        f = fake_finding(b.researcher, n_kept=2)
        run.findings.append(f)
        return f

    monkeypatch.setattr(research, "run_researcher", fake_researcher)
    run = new_run()
    first = asyncio.run(research.dispatch(run, research.default_briefs(ITEM)))
    assert peak == 3  # all three researchers ran at the same time
    assert first["round"] == 1 and any("Only 2 exact comps" in g for g in first["gaps"])
    asyncio.run(research.dispatch(run, [brief(broadened=True)]))
    asyncio.run(research.dispatch(run, [brief(broadened=True)]))
    assert asyncio.run(research.dispatch(run, [brief()])) == {"error": "No research rounds left. Reply DONE."}
    assert run.rounds == research.MAX_ROUNDS


# --- Confidence and citations ---


def run_with_sources(n_comps, broadened=False, retail=True, retail_price=249.95, successor=False, prices=None):
    run = new_run()
    prices = prices or [80.0 + i for i in range(1, n_comps + 1)]
    for i, price in enumerate(prices, 1):
        run.register("E", str(i), {"title": f"listing {i}", "price": price, "shippingCost": 0.0, "totalPrice": price, "condition": "Used",
                                   "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": f"https://www.ebay.com/itm/{i}"})
    run.findings.append(fake_finding(n_kept=len(prices), broadened=broadened))
    if retail:
        run.register("R", "p1", {"title": "AirPods Pro 2", "retailer": "Best Buy", "price": f"${retail_price}", "extracted_price": retail_price,
                                 "old_price": None, "extracted_old_price": None, "link": "https://example.com/r1"})
        run.findings.append(fake_finding("retail", n_kept=1, kept_prefix="R", successor=successor))
    return run


def test_confidence_cap_from_comp_count_and_spread():
    assert research.confidence_cap(run_with_sources(3))[0] == "low"
    assert research.confidence_cap(run_with_sources(7))[0] == "medium"
    assert research.confidence_cap(run_with_sources(12, broadened=True))[0] == "medium"
    assert research.confidence_cap(run_with_sources(12))[0] == "high"
    # no retail price no longer lowers confidence: count and spread decide
    assert research.confidence_cap(run_with_sources(12, retail=False))[0] == "high"
    medium_spread = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]  # IQR 55 / median 115
    assert research.confidence_cap(run_with_sources(12, prices=medium_spread)) [0] == "medium"
    wide = [20, 30, 40, 60, 80, 100, 120, 150, 180, 200, 220, 250]
    level, reason = research.confidence_cap(run_with_sources(12, prices=wide))
    assert level == "low" and "vary widely" in reason


def test_successor_ceiling_caps_confidence_at_medium():
    level, reason = research.confidence_cap(run_with_sources(12, successor=True))
    assert level == "medium" and "successor" in reason


def test_retail_reference_is_a_ceiling_below_retail_and_a_premium_above():
    ceiling = research.retail_reference(run_with_sources(12))
    assert ceiling["role"] == "ceiling" and ceiling["resale_to_retail_ratio"] == round(86.5 / 249.95, 2)
    premium = research.retail_reference(run_with_sources(12, retail_price=60.0))
    assert premium["role"] == "premium" and premium["resale_to_retail_ratio"] == round(86.5 / 60, 2)
    # a premium is no ceiling: pricing above retail doesn't warn
    run = run_with_sources(12, retail_price=60.0)
    memo = Memo(price_low=82, price_high=88, headline="h", rationale="r", key_factors=[], confidence="high", confidence_reason="c", limitations=[])
    assert not any("retail" in w for w in research.validate_memo(run, memo, research.confidence_cap(run))[1]["warnings"])
    markdown, _ = research.render(run, {**memo.model_dump()}, {"comps": research.comp_rows(run), "tiers": [], "retail": research.retail_rows(run)}, [])
    assert "Resale premium: comps ask 1.44x" in markdown


def test_own_retail_price_wins_over_the_successors():
    run = run_with_sources(12, successor=True)
    run.register("R", "p2", {"title": "AirPods Pro 2", "retailer": "Target", "price": "$229", "extracted_price": 229.0,
                             "old_price": None, "extracted_old_price": None, "link": "https://example.com/r2"})
    own = fake_finding("retail", n_kept=0)
    own["kept_ids"], own["n_kept"] = ["R2"], 1
    run.findings.append(own)
    ref = research.retail_reference(run)
    assert ref["id"] == "R2" and ref["successor"] is False


def test_check_citations_keeps_real_tags_and_strips_invented_ones():
    sources = {"E1": {}, "E2": {}, "R1": {}}
    text, cited, unknown = research.check_citations("Median $80 [E1, E2], new $249 [R1], also $5 [E77]. Model A2698, see E2 and E9.", sources)
    assert text == "Median $80 [E1][E2], new $249 [R1], also $5 . Model A2698, see [E2] and E9."
    assert cited == ["E1", "E2", "R1", "E2"]
    assert unknown == ["E77"]


def test_validate_memo_caps_confidence_sorts_range_and_warns_above_retail():
    run = run_with_sources(3)
    memo = Memo(price_low=260, price_high=90, headline="List at $90 [E1] [E99]", rationale="r [R1]", key_factors=["E2 at $82"],
                confidence="high", confidence_reason="lots of data", limitations=["asking prices are not sold prices"])
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert (clean["price_low"], clean["price_high"]) == (90, 260)
    assert clean["confidence"] == "low" and "Capped" in clean["confidence_reason"]
    assert clean["headline"] == "List at $90 [E1] " and clean["key_factors"] == ["[E2] at $82"]
    assert check["unknown_tags_stripped"] == ["E99"] and set(check["cited"]) == {"E1", "E2", "R1"}
    assert any("lowest new retail price" in w for w in check["warnings"])


def test_no_comps_says_so_instead_of_guessing():
    run = new_run()
    run.findings.append(fake_finding(n_kept=0))
    response, memo = asyncio.run(research.write_memo(run))
    assert memo is None and "couldn't find enough" in response


def test_render_cites_only_real_sources_and_reports_the_funnel():
    run = run_with_sources(20)
    memo = {"price_low": 80, "price_high": 90, "headline": "h", "confidence": "high", "confidence_reason": "c", "rationale": "r [E3]",
            "key_factors": ["k"], "limitations": ["l"]}
    rows = {"comps": research.comp_rows(run), "tiers": [], "retail": research.retail_rows(run)}
    markdown, sources = research.render(run, memo, rows, ["E3"])
    assert "20 found, 20 after the title filter, 20 after review" in markdown
    assert markdown.count("| [E") == research.TABLE_ROWS  # an even spread, not all 20
    assert set(sources) >= {"E3", "R1"} and all(s in run.sources for s in sources)
    assert "not a prediction of what it will sell for" in markdown


# --- Variant check ---


@pytest.mark.parametrize("title, reason", [
    ("Apple iPhone 13 Pro 128GB Unlocked Sierra Blue", None),
    ("iPhone 13 Pro 128 GB unlocked", None),
    ("Apple iPhone 13 Pro", "variant not stated: 128gb"),
    ("Apple iPhone 13 Pro 256GB Unlocked", "different variant: 256gb"),
    ("iPhone 13 Pro 128GB 256GB 512GB Unlocked", "several variants in one listing"),
    ("Apple iPhone 13 Pro 128GB Gold - Verizon", "variant not stated: unlocked"),
])
def test_variant_check_for_storage_and_carrier(title, reason):
    assert research.variant_exclusion(title, "128GB, unlocked") == reason


def test_variant_check_ignores_ram_and_items_without_a_variant():
    assert research.variant_exclusion("MacBook Air M2 8GB RAM 256GB SSD", "256GB") is None
    assert research.variant_exclusion("AirPods Pro 2", "") is None


def test_mostly_resale_retail_results_count_as_no_new_retail(monkeypatch):
    offer = {"title": "AirPods Pro 2", "price": "$90", "extracted_price": 90.0, "old_price": None, "extracted_old_price": None, "second_hand_condition": None}
    items = [{**offer, "retailer": "eBay", "is_resale": True, "link": f"https://e/{i}", "product_id": str(i)} for i in range(9)]
    items.append({**offer, "retailer": "Best Buy", "is_resale": False, "link": "https://bb/1", "product_id": "bb"})
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": items})
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2"))
    research.do_review(state, [Exclusion(id=next(iter(state.shown)), reason="different model")])
    assert state.finding["only_resale_listings"] is True  # 9 of 10 resale, the one store offer excluded
    research.do_review(state, [])
    state2 = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state2, "AirPods Pro 2"))
    research.do_review(state2, [])
    assert state2.finding["only_resale_listings"] is False  # a store offer was kept


@pytest.mark.parametrize("title, product, reason", [
    ("Apple iPhone 13 Pro 128GB Unlocked", "Apple iPhone 13 Pro", None),
    ("Apple iPhone 13 Pro Max 128GB Unlocked", "Apple iPhone 13 Pro", "different model: Max"),
    ("Apple iPhone 13 - 128GB - Blue (Unlocked)", "Apple iPhone 13 Pro", "different model: not Pro"),
    ("Apple iPhone 14 Plus 128GB", "Apple iPhone 14", "different model: Plus"),
    ("AirPods Pro 2 plus extra ear tips", "Apple AirPods Pro (2nd generation)", None),
    ("Apple AirPods Pro 2nd Gen USB-C", "Apple AirPods Pro (2nd generation)", None),
])
def test_model_tier_check(title, product, reason):
    assert research.model_exclusion(title, product) == reason


@pytest.mark.parametrize("text, expected", [
    ("listings like [E18] ($80.00 [E18]) and [E20] ($84.99 [E20])", "listings like [E18] ($80.00) and [E20] ($84.99)"),
    ("compared to [E70] at $299.99 [E70].", "compared to [E70] at $299.99."),
    ("median $85 [E1][E1]", "median $85 [E1]"),
    ("from $80 [E1] to $90 [E2] and back [E1]", "from $80 [E1] to $90 [E2] and back [E1]"),  # another tag in between
    ("low at $80 [E1]. The next sentence repeats it [E1].", "low at $80 [E1]. The next sentence repeats it [E1]."),
])
def test_adjacent_repeated_citations_are_cited_once(text, expected):
    assert research.dedupe_citations(text) == expected


def test_check_citations_dedupes_after_bracketing_bare_ids():
    text, cited, _ = research.check_citations("listings like E18 ($80.00 [E18])", {"E18": {}})
    assert text == "listings like [E18] ($80.00)"
