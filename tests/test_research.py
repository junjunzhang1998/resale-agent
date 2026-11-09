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
    memo = Memo(price_low=260, price_high=90, headline="List at $81 [E1] [E99]", rationale="r [R1]", key_factors=["E2 at $82"],
                confidence="high", confidence_reason="lots of data", limitations=["asking prices are not sold prices"])
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert (clean["price_low"], clean["price_high"]) == (90, 260)
    assert clean["confidence"] == "low" and "Capped" in clean["confidence_reason"]
    assert clean["headline"] == "List at $81 [E1] " and clean["key_factors"] == ["[E2] at $82"]
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


# --- Special editions ---


@pytest.mark.parametrize("title, item_text, reason", [
    ("Nike Air Force 1 Low White size 10", "Nike Air Force 1 Low size 10", None),
    ("Nike Air Force 1 Low FlyEase White size 10", "Nike Air Force 1 Low size 10", "special edition: FlyEase"),
    ("Nike Air Force 1 x Off-White size 10", "Nike Air Force 1 Low size 10", "special edition: collab"),
    ("Custom painted Nike Air Force 1 size 10", "Nike Air Force 1 Low size 10", "special edition: custom / painted"),
    ("Apple AirPods Pro 2 ColorWare Matte Black", "Apple AirPods Pro 2", "special edition: custom / painted"),
    ("Apple Watch SE 44mm", "Apple Watch Series 9 44mm", "special edition: SE edition"),
    ("Apple iPhone SE 64GB", "Apple iPhone SE 64GB", None),  # the seller named the SE
    ("Nintendo Switch OLED Zelda Limited Edition", "Nintendo Switch OLED", "special edition: special edition"),
])
def test_special_editions_are_dropped_unless_named(title, item_text, reason):
    assert research.edition_exclusion(title, item_text) == reason


# --- One comp count ---


def test_writer_gets_one_merged_comp_count():
    run = run_with_sources(12)
    second = fake_finding(n_kept=3, broadened=True)  # a later round re-finding E1..E3
    run.findings.append(second)
    data = json.loads(research.writer_input(run, research.confidence_cap(run)))
    assert data["comps"]["kept"] == 12 and data["comps"]["stats_n"] == 12
    assert all(f["researcher"] != "exact_comps" for f in data["findings"])  # no per-round counts to quote


def test_count_mismatches_flags_other_counts_but_not_subsets():
    texts = ["Median $85 across 19 of 27 comps that list shipping.", "Based on 27 kept comps.", "Excluded 4 listings (case only).",
             "Just under the median of 44 comparable listings."]
    assert research.count_mismatches(texts, 27) == [44]


def test_condition_tiers_dedupe_and_order():
    run = new_run()
    tiers = lambda: {t: {"price_with_shipping": {"n": 2, "median": m, "q1": m, "q3": m}, "item_price": {"n": 2}} for t, m in
                     [("used", 80), ("new", 150), ("for_parts", 40)]}
    for _ in range(2):
        f = fake_finding("condition_tiers")
        f["stats_by_condition"] = tiers()
        run.findings.append(f)
    assert [r["condition"] for r in research.tier_rows(run)] == ["new", "used", "for_parts"]


# --- Draft my listing ---


MEMO = {"item": {"product": "Apple AirPods Pro (2nd generation)", "variant": "", "condition": "used", "condition_detail": "good",
                 "included": "charging case"}, "assumptions": [], "price_low": 78.0, "price_high": 85.0,
        "comps": [{"shipping": 0}, {"shipping": 0}, {"shipping": 5.0}]}


def test_clip_title_respects_ebays_80_characters():
    long = "Apple iPhone 13 Pro 128GB Sierra Blue Unlocked Smartphone with Original Box and Charger Cable - Used"
    clipped = research.clip_title(long)
    assert len(clipped) <= research.EBAY_TITLE_MAX and long.startswith(clipped) and not clipped.endswith(" ")


def test_draft_falls_back_to_the_template_and_invents_nothing(monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(research.Runner, "run", broken)
    draft = asyncio.run(research.draft_listing(MEMO))
    assert draft["source"] == "template" and draft["price"] == 85.0
    assert draft["title"] == "Apple AirPods Pro (2nd generation) with charging case - Used"
    assert "Condition: good" in draft["description"] and "Free shipping." in draft["description"]


def test_draft_cleanup_drops_title_placeholders_and_splits_bullets():
    assert research.clip_title("Apple AirPods Pro 2nd Gen with [USB-C or Lightning] Charging Case - Used") == "Apple AirPods Pro 2nd Gen with Charging Case - Used"
    text = "Up for sale. - Condition: Good. [add photos] - Included: Case - Shipping: Free."
    assert research.tidy_description(text) == "Up for sale.\n- Condition: Good. [add photos]\n- Included: Case\n- Shipping: Free."


def test_writer_is_asked_once_more_when_its_comp_counts_are_off(monkeypatch):
    run = run_with_sources(12)
    good = Memo(price_low=82, price_high=88, headline="List at $82-$88 from 12 kept comps [E3]", rationale="r", key_factors=[],
                confidence="high", confidence_reason="c", limitations=["asking prices are not sold prices"])
    bad = good.model_copy(update={"headline": "List at $82-$88 from 44 comparable listings [E3]"})
    prompts = []

    class Result:
        def __init__(self, out):
            self.final_output = out

    async def fake_run(agent, prompt, **kw):
        prompts.append(prompt)
        return Result(bad if len(prompts) == 1 else good)

    monkeypatch.setattr(research.Runner, "run", fake_run)
    _, memo = asyncio.run(research.write_memo(run))
    assert len(prompts) == 2 and "there are 12 kept comps" in prompts[1]
    assert "12 kept comps" in memo["headline"] and not memo["citation_check"]["warnings"]
    assert any(c["name"] == "recount" for c in run.trace.calls)


def test_sellers_condition_uses_the_exact_comps_median_everywhere():
    run = run_with_sources(12)  # exact comps: 81..92, median 86.5
    f = fake_finding("condition_tiers")
    f["stats_by_condition"] = {t: {"price_with_shipping": {"n": 5, "median": m, "q1": m, "q3": m}, "item_price": {"n": 5}}
                               for t, m in [("new", 150.0), ("used", 99.0)]}
    run.findings.append(f)
    rows = {r["condition"]: r for r in research.tier_rows(run)}
    assert rows["used"]["median"] == 86.5 and rows["used"]["source"] == "exact_comps" and rows["used"]["n"] == 12
    assert rows["new"]["source"] == "condition_search"
    data = json.loads(research.writer_input(run, research.confidence_cap(run)))
    tiers = next(x for x in data["findings"] if x["researcher"] == "condition_tiers")
    assert "used" not in tiers["stats_by_condition"] and "stats" not in tiers  # no second used median to quote


def test_draft_description_gets_real_line_breaks():
    literal = "Intro.\\n- Condition: Good.\\n- Included: Case"  # backslash-n as two characters
    assert "\n" not in literal
    assert research.tidy_description(literal) == "Intro.\n- Condition: Good.\n- Included: Case"


# --- Category-aware exclusions (Chanel "Single Flap" is not a single earbud) ---

CHANEL = Item(product="Chanel Classic Flap", variant="medium, caviar, gold hardware", condition="used",
              condition_detail="good condition, light corner wear", included="dust bag", search_query="Chanel Classic Flap medium caviar")


def chanel_run():
    return Run(item=CHANEL, assumptions=[], trace=Trace())


def test_item_profile_picks_category_luxury_and_boutique_brand():
    assert research.item_profile(ITEM) == {"category": "earbuds", "luxury": False, "boutique_brand": None}
    assert research.item_profile(CHANEL) == {"category": "handbag", "luxury": True, "boutique_brand": "Chanel"}
    phone = ITEM.model_copy(update={"product": "Apple iPhone 13 Pro", "search_query": "iPhone 13 Pro 128GB"})
    assert research.item_profile(phone)["category"] == "phone"


@pytest.mark.parametrize("title, reason", [
    ("CHANEL Classic Single Flap Bag Medium Black Caviar GHW", None),  # a real bag, not an earbud
    ("Chanel Classic Flap Medium Caviar Gold Hardware Style A01112", None),  # a style number
    ("Chanel Classic Flap Inspired Quilted Bag", "not genuine"),
    ("Quilted CC Style Flap Bag Black", "not genuine"),
    ("Chanel look alike quilted flap", "not genuine"),
    ("Classic Flap bag NOT AUTHENTIC dupe", "not genuine"),
    ("Purse organizer insert for Chanel Classic Flap Medium", "accessory, not the bag"),
    ("Chanel dust bag large", "dust bag / box only"),
    ("Chanel Classic Flap Medium Caviar with dust bag and box", None),
])
def test_handbag_title_filter(title, reason):
    # the seller's own "dust bag" is what's included, not the item: check against the product only
    assert research.title_exclusion(title, f"{CHANEL.product} {CHANEL.variant}", category="handbag") == reason


def test_single_still_means_one_earbud_for_earbuds():
    assert research.title_exclusion("Single AirPods Pro 2 earbud", category="earbuds") == "single earbud / one side"
    assert research.title_exclusion("Single AirPods Pro 2 earbud", category="handbag") is None


@pytest.mark.parametrize("title, reason", [
    ("Chanel Classic Flap Medium Black Caviar GHW", None),
    ("Chanel Classic Medium/Large Flap Caviar Gold", None),  # M/L is the medium
    ("Chanel Classic Flap Jumbo Black Caviar GHW", "different size: jumbo"),
    ("Chanel Classic Flap Medium Lambskin Gold Hardware", "different leather: lambskin"),
    ("Chanel Classic Flap Medium Caviar Silver Hardware", "different hardware: silver"),
    ("Chanel Classic Flap Black Caviar", None),  # size not stated: left to the review
])
def test_bag_variant_check(title, reason):
    assert research.bag_variant_exclusion(title, "Chanel Classic Flap medium, caviar, gold hardware") == reason


def test_reviewer_reasons_are_kept_in_its_own_words():
    assert research.review_reason("  Different style: Single Flap, not Double.  ") == "different style: Single Flap, not Double"
    assert research.review_reason("") == "not the exact item"
    assert len(research.review_reason("x" * 100)) <= 60


# --- Luxury retail ---


def test_luxury_resale_platforms_are_resale():
    for source in ("Fashionphile", "The RealReal", "Rebag", "Vestiaire Collective", "Yoogi's Closet"):
        assert market.retail_record({"source": source, "title": "Chanel Classic Flap"})["is_resale"], source


def test_boutique_only_brand_has_no_retail_gap_and_says_not_sold_online(monkeypatch):
    offers = [{"title": "Chanel Classic Flap Medium", "retailer": r, "is_resale": False, "link": f"https://s/{i}", "product_id": str(i),
               "price": "$9,000", "extracted_price": 9000.0, "old_price": None, "extracted_old_price": None, "second_hand_condition": None}
              for i, r in enumerate(["Madison Ave Luxe", "Some Boutique"])]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": offers})
    run = chanel_run()
    state = ResearcherState(run, brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "Chanel Classic Flap"))
    research.do_review(state, [])
    run.findings.append(state.finding)
    assert state.finding["n_kept"] == 0 and all("boutiques" in d["reason"] for d in state.dropped)
    assert research.find_gaps(run.findings, CHANEL) == []
    status = research.retail_status(run)
    assert status["status"] == "boutique_only" and "discontinued" not in status["note"] and "boutiques" in status["note"]


def test_retail_status_discontinued_vs_not_found():
    run = new_run()
    run.findings.append(fake_finding("retail", n_kept=0, only_resale=True))
    assert research.retail_status(run)["status"] == "discontinued"
    run = new_run()
    run.findings.append(fake_finding("retail", n_kept=0))
    assert research.retail_status(run)["status"] == "not_found"


def test_luxury_memo_gets_the_authenticity_limitation_and_no_authenticity_assumption():
    assert research.split_assumptions(["Color not given: comps include every color. Authenticity assumed genuine.", "Size: medium"]) == \
        ["Color not given: comps include every color.", "Size: medium"]
    run = chanel_run()
    for i, price in enumerate([5000.0 + 100 * i for i in range(8)], 1):
        run.register("E", str(i), {"title": f"Chanel {i}", "price": price, "shippingCost": 0.0, "totalPrice": price, "condition": "Pre-owned - Good",
                                   "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": f"https://www.ebay.com/itm/{i}"})
    run.findings.append(fake_finding(n_kept=8))
    memo = Memo(price_low=5000, price_high=5400, headline="h", rationale="r.", key_factors=[], confidence="medium", confidence_reason="c",
                limitations=["Asking prices are not sold prices.", "Listings may not be authentic."])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["limitations"] == ["Asking prices are not sold prices.", research.AUTH_LIMITATION]


# --- One funnel, one exclusion sentence ---


def test_funnel_counts_each_listing_once_and_adds_up():
    run = run_with_sources(10)
    first = run.findings[0]
    first["excluded"] = [{"id": "E90", "title": "x", "reason": "case only", "by": "code"},
                         {"id": "E91", "title": "y", "reason": "different model: Pro 1", "by": "review"},
                         {"id": "E92", "title": "z", "reason": "different model: Max", "by": "review"}]
    again = fake_finding(n_kept=3, broadened=True)  # round 2 re-finds E1..E3 and re-excludes E90
    again["excluded"] = [{"id": "E90", "title": "x", "reason": "case only", "by": "code"}]
    run.findings.append(again)
    f = research.comp_funnel(run)
    assert (f["found"], f["after_title_filter"], f["kept"], f["excluded"]) == (13, 12, 10, 3)
    assert f["found"] == f["kept"] + f["excluded"]
    assert f["excluded_groups"] == [{"label": "different model", "n": 2, "details": ["1× Pro 1", "1× Max"]},
                                    {"label": "case only", "n": 1, "details": []}]
    assert research.exclusion_sentence(f) == "Excluded 3 of 13 listings: 2 different model and 1 case only."


def test_writers_own_exclusion_counts_are_replaced_with_codes():
    run = run_with_sources(10)
    run.findings[0]["excluded"] = [{"id": "E90", "title": "x", "reason": "single earbud / one side", "by": "code"}]
    memo = Memo(price_low=82, price_high=88, headline="h", confidence="high", confidence_reason="c",
                rationale="Comps sit near $83 [E3]. Excluded 4 listings (3 single earbuds, 1 case only). Prices dropped 20% this month.",
                key_factors=["3 single earbuds were removed", "Free shipping is common"], limitations=["Asking prices are not sold prices."])
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["rationale"] == ("Comps sit near $83 [E3]. Prices dropped 20% this month. "
                                  "Excluded 1 of 11 listings: 1 single earbud / one side.")
    assert clean["key_factors"] == ["Free shipping is common"]
    assert any("exclusion counts were replaced" in w for w in check["warnings"])


def test_condition_row_names_its_subset_when_shipping_is_missing():
    run = run_with_sources(10)
    for sid in ("E1", "E2", "E3"):  # three comps with shipping not stated
        run.sources[sid].update(shippingCost=None, totalPrice=None)
    cm = research.comp_median(run)
    assert (cm["basis"], cm["n"], cm["of"]) == ("price_with_shipping", 7, 10)
    assert "across the 7 of 10 comps that list shipping" in cm["phrase"]
    row = research.tier_rows(run)[0]
    assert (row["n"], row["of"], row["median"]) == (7, 10, cm["median"])
    assert [r["in_median"] for r in research.comp_rows(run)].count(False) == 3


# --- Condition sub-grades ---


@pytest.mark.parametrize("condition, grade", [
    ("Pre-owned - Excellent", "Excellent"), ("Pre-owned - Good", "Good"), ("Pre-owned - Fair", "Fair"),
    ("Pre-owned - Poor", "Poor"), ("Used", None), ("Pre-owned", None), ("Like New", "Excellent"),
])
def test_listing_grade(condition, grade):
    assert research.listing_grade(condition) == grade


@pytest.mark.parametrize("detail, grade", [
    ("good, light scratches on case", "Good"), ("light scratches", "Good"), ("works fine, some wear", "Good"),
    ("excellent, barely used", "Excellent"), ("very good", "Very Good"), ("heavy wear on corners", "Fair"), ("", None),
])
def test_seller_grade_from_their_words(detail, grade):
    assert research.seller_grade(ITEM.model_copy(update={"condition_detail": detail})) == grade


def test_weighted_median():
    assert market.weighted_median([10, 20, 30], [1, 1, 1]) == 20
    assert market.weighted_median([10, 20, 30, 100], [0.2, 0.2, 1, 1]) == 30
    assert market.weighted_median([10, 20], [1, 1]) == 15
    assert market.weighted_median([], []) is None


def test_comps_are_weighted_toward_the_sellers_grade_and_both_medians_reported():
    grades = ["Pre-owned - Excellent"] * 6 + ["Pre-owned - Good"] * 4 + ["Used"] * 2
    prices = [120.0, 121, 122, 123, 124, 125, 90, 92, 94, 96, 110, 112]
    run = run_with_sources(12, prices=prices)
    for i, g in enumerate(grades, 1):
        run.sources[f"E{i}"]["condition"] = g
    g = research.grade_summary(run)  # seller: "good"
    assert g["grade"] == "Good" and g["n_matching"] == 4 and g["n"] == 12
    assert g["matching_median"] == 93.0 and g["used_median"] == 116.0 and g["differs"]
    assert g["weighted_median"] < g["used_median"]
    assert "Used median $116.00 across all 12 comps" in g["note"] and "Pre-owned - Good (your grade) ask a median $93.00" in g["note"]
    assert research.anchor_median(run) == g["weighted_median"]
    assert sum(r["grade_match"] for r in research.comp_rows(run)) == 4  # every comp still listed
    memo = Memo(price_low=90, price_high=100, headline="h", rationale="r.", key_factors=["k"], confidence="high", confidence_reason="c", limitations=[])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["key_factors"][0] == research.check_citations(g["note"], run.sources)[0] and clean["key_factors"][1] == "k"


# --- Sealed new far below retail ---


def test_new_listings_far_below_retail_become_a_key_factor():
    run = run_with_sources(12)  # retail $249.95 at Best Buy [R1]
    for i, price in enumerate([125.0, 130, 135, 140], 100):
        run.register("E", str(i), {"title": f"new {i}", "price": price, "shippingCost": 0.0, "totalPrice": price, "condition": "New",
                                   "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": f"https://www.ebay.com/itm/{i}"})
    tiers = fake_finding("condition_tiers")
    tiers["tier_ids"] = {"new": ["E13", "E14", "E15", "E16"]}
    run.findings.append(tiers)
    gap = research.new_vs_retail(run)
    assert gap["median"] == 132.5 and gap["ratio"] == round(132.5 / 249.95, 2)
    assert "47% below the $249.95 new price at Best Buy [R1]" in gap["sentence"]
    memo = Memo(price_low=82, price_high=88, headline="h", rationale="r.", key_factors=["k"], confidence="high", confidence_reason="c", limitations=[])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["key_factors"][0].startswith("Sealed new listings on eBay ask a median $132.50")
    tiers["tier_ids"] = {"new": ["E13"]}  # too few new listings to say
    assert research.new_vs_retail(run) is None


# --- Retail titles ---


def new_below_retail_run():
    run = run_with_sources(12)  # comps 81..92, retail $249.95 at Best Buy [R1]
    for i, price in enumerate([125.0, 130, 135, 140], 100):
        run.register("E", str(i), {"title": f"new {i}", "price": price, "shippingCost": 0.0, "totalPrice": price, "condition": "New",
                                   "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": f"https://www.ebay.com/itm/{i}"})
    tiers = fake_finding("condition_tiers")
    tiers["tier_ids"] = {"new": ["E13", "E14", "E15", "E16"]}
    run.findings.append(tiers)
    return run


def memo_with(**kw):
    base = dict(price_low=82, price_high=88, headline="h", rationale="r.", key_factors=[], confidence="high", confidence_reason="c", limitations=[])
    return Memo(**{**base, **kw})


# --- Key factors appear once ---


def test_key_factors_are_deduped_against_codes_own():
    run = new_below_retail_run()
    gap = research.new_vs_retail(run)["sentence"]
    memo = memo_with(key_factors=[gap, "Sealed new units on eBay sell well below the $249.95 retail price [R1].",
                                  "Free shipping is standard.", "Free shipping is standard!"])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["key_factors"] == [gap, "Free shipping is standard."]


# --- Stats are cited to their basis, listings only for their own price ---


def test_a_stat_is_cited_to_its_basis_not_a_listing():
    grades = ["Pre-owned - Excellent"] * 6 + ["Pre-owned - Good"] * 4 + ["Used"] * 2
    run = run_with_sources(12, prices=[120.0, 121, 122, 123, 124, 125, 90, 92, 94, 96, 110, 112])
    for i, g in enumerate(grades, 1):
        run.sources[f"E{i}"]["condition"] = g
    w = research.grade_summary(run)["weighted_median"]
    # even a listing whose total equals the stat (as E3's did for Chanel) isn't its source
    holder = next(s for s in run.sources if s.startswith("E") and run.sources[s]["totalPrice"] == w)
    text = f"List just below the anchor median of ${w:,.2f} [{holder}]."
    assert research.fix_stat_citations(text, run) == f"List just below the anchor median of ${w:,.2f} (grade-weighted median of 12 comps)."
    # a basis already named: the tag just goes
    assert research.fix_stat_citations("the median $116.00 across all 12 comps [E1][E2].", run) == "the median $116.00 across all 12 comps."
    # a quartile with no basis gets one
    q1 = research.comp_median(run)["q1"]
    assert research.fix_stat_citations(f"near the Q1 of ${q1:.2f} [E9]", run) == f"near the Q1 of ${q1:.2f} (Q1 of all 12 comps)"


def test_a_listing_tag_stays_only_on_that_listings_own_price():
    run = run_with_sources(12)  # E1 = $81 with free shipping, E2 = $82
    assert research.fix_stat_citations("List at $81 [E1] to $90 [E2].", run) == "List at $81 [E1] to $90."
    assert research.fix_stat_citations("from $82 [E1][E2]", run) == "from $82 [E2]"
    assert research.fix_stat_citations("new at Best Buy for $249.95 [R1], was ignored", run) == "new at Best Buy for $249.95 [R1], was ignored"
    assert research.fix_stat_citations("see [E1] and [E2]", run) == "see [E1] and [E2]"  # no amount: untouched


def test_validated_memo_lists_only_the_citations_that_survive():
    run = run_with_sources(12)
    clean, check = research.validate_memo(run, memo_with(headline="List at $81 [E1], near the median $86.50 [E5][E6]."), research.confidence_cap(run))
    assert "[E5]" not in clean["headline"] and check["cited"] == ["E1"]


# --- Best Offer only when offers are common ---


def test_best_offer_is_a_factor_only_when_30_percent_of_comps_take_offers():
    run = run_with_sources(10)
    memo = memo_with(rationale="Comps cluster near $86. Enable Best Offer to catch buyers.", key_factors=["Best Offer gives room to haggle", "Free shipping"])
    for sid in ("E1", "E2"):  # 2 of 10 take offers
        run.sources[sid]["buyingOptions"] = ["FIXED_PRICE", "BEST_OFFER"]
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["key_factors"] == ["Free shipping"] and "Best Offer" not in clean["rationale"]
    run.sources["E3"]["buyingOptions"] = ["FIXED_PRICE", "BEST_OFFER"]  # 3 of 10: 30%
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["key_factors"][0] == "Best Offer gives room to haggle"
    assert research.offers_common({"comps": [{"best_offer": True}, {"best_offer": False}, {"best_offer": False}, {"best_offer": False}]}) is False


def test_retail_title_names_the_model_when_the_store_leaves_it_out():
    title, note = research.retail_title("Personalized AirPods Pro", "Apple AirPods Pro (2nd generation)")
    assert title == "AirPods Pro" and "personalization option" in note and "doesn't state the model" in note
    assert research.retail_title("Apple AirPods Pro 2 with USB-C", "Apple AirPods Pro (2nd generation)") == ("Apple AirPods Pro 2 with USB-C", "")
    assert research.retail_title("Apple iPhone 13 Pro 128GB", "Apple iPhone 13 Pro")[1] == ""
