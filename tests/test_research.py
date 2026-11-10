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


def test_only_resale_retail_results_get_one_retry_then_are_final(monkeypatch):
    resale = [{"title": "AirPods Pro 2", "retailer": "eBay", "is_resale": True, "link": f"https://e/{i}", "product_id": str(i),
               "price": "$90", "extracted_price": 90.0, "old_price": None, "extracted_old_price": None, "second_hand_condition": "used"} for i in range(3)]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": resale})
    state = ResearcherState(new_run(), brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "AirPods Pro 2"))
    research.do_review(state, [])
    assert state.finding["only_resale_listings"] is True and state.finding["n_kept"] == 0
    # one short query: try once more with the store's full name ("Dyson V8" found Dyson's own store that way)
    [gap] = research.find_gaps([fake_finding(), state.finding])
    assert "Try once more with the full product name" in gap and "not a successor" in gap
    asyncio.run(research.do_search_retail(state, "Apple AirPods Pro 2nd Generation Wireless Earbuds"))
    research.do_review(state, [])
    assert research.find_gaps([fake_finding(), state.finding]) == []  # two queries: final
    state.run.findings.append(state.finding)
    assert research.no_new_retail(state.run)


def test_comps_mixing_product_lines_cap_confidence_at_medium():
    run = run_with_sources(12)
    run.item = ITEM.model_copy(update={"product": "Dyson V8", "search_query": "Dyson V8"})
    for sid, line in (("E1", "Animal"), ("E2", "Absolute"), ("E3", "Origin")):
        run.sources[sid]["title"] = f"Dyson V8 {line} Cordless Vacuum"
    assert research.mixed_lines(run) == ["absolute", "animal", "origin"]
    level, reason = research.confidence_cap(run)
    assert level == "medium" and "mix product lines" in reason
    run.item = run.item.model_copy(update={"product": "Dyson V8 Animal"})  # the seller named the line: no mix
    assert research.mixed_lines(run) == []


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
    memo = Memo(price_low=82, price_high=88, headline="h", why=[], watch_out=[], confidence="high", confidence_reason="c")
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert not any("tops the retail ceiling" in w for w in check["warnings"])
    # ...but for an everyday used item, asking above retail is flagged and caps confidence at medium
    assert clean["confidence"] == "medium" and clean["watch_out"][0].startswith("Used listings ask more than retail")
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
    memo = Memo(price_low=260, price_high=90, headline="List at $81 [E1] [E99]", why=["E2 at $82", "r [R1]"],
                confidence="high", confidence_reason="lots of data", watch_out=["asking prices are not sold prices"])
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert (clean["price_low"], clean["price_high"]) == (90, 260)
    assert clean["confidence"] == "low" and clean["confidence_reason"].startswith("Only 3 comparable listings")  # the cap, in one sentence
    assert clean["headline"] == "List at $81 [E1]" and clean["why"][1:] == ["[E2] at $82", "r [R1]"]
    assert clean["watch_out"] == []  # a generic caveat: the page says it once, under the price
    assert check["unknown_tags_stripped"] == ["E99"] and set(check["cited"]) >= {"E1", "E2", "R1"}
    assert any("tops the retail ceiling" in w for w in check["warnings"])


def test_no_comps_says_so_instead_of_guessing():
    run = new_run()
    run.findings.append(fake_finding(n_kept=0))
    response, memo = asyncio.run(research.write_memo(run))
    assert memo is None and "couldn't find enough" in response


def test_render_cites_only_real_sources_and_reports_the_funnel():
    run = run_with_sources(20)
    memo = {"price_low": 80, "price_high": 90, "headline": "h", "confidence": "high", "confidence_reason": "c", "why": ["r [E3]", "k"],
            "watch_out": ["l"]}
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
    good = Memo(price_low=82, price_high=88, headline="List at $82-$88 from 12 kept comps [E3]", why=[],
                confidence="high", confidence_reason="c", watch_out=[])
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
    assert research.item_profile(ITEM) == {"category": "earbuds", "luxury": False, "boutique_brand": None, "similar_items": False}
    assert research.item_profile(CHANEL) == {"category": "handbag", "luxury": True, "boutique_brand": "Chanel", "similar_items": False}
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
    memo = Memo(price_low=5000, price_high=5400, headline="h", why=[], confidence="medium", confidence_reason="c",
                watch_out=["Asking prices are not sold prices.", "Listings may not be authentic.", "Chanel sizes are in centimeters: check yours."])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["watch_out"] == [research.AUTH_LIMITATION, "Chanel sizes are in centimeters: check yours."]


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


def test_exclusions_stay_out_of_the_memo_prose():
    run = run_with_sources(10)
    run.findings[0]["excluded"] = [{"id": "E90", "title": "x", "reason": "single earbud / one side", "by": "code"}]
    memo = Memo(price_low=82, price_high=88, headline="h", confidence="high", confidence_reason="c",
                why=["Comps sit near $83 [E3].", "3 single earbuds were removed", "Free shipping is common"],
                watch_out=["Excluded 4 listings (3 single earbuds, 1 case only).", "Prices dropped 20% this month."])
    clean, check = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["why"][1:] == ["Comps sit near $83 [E3].", "Free shipping is common"]
    assert clean["watch_out"] == ["Prices dropped 20% this month."]  # a percentage isn't an exclusion count
    assert not any("xclu" in t or "removed" in t for t in [clean["headline"], *clean["why"], *clean["watch_out"]])


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
    memo = Memo(price_low=90, price_high=100, headline="h", why=["k"], confidence="high", confidence_reason="c", watch_out=[])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["why"][1] == "Your grade matters: Pre-owned - Good listings ask a median $93, vs $116 for all used comps."
    assert clean["why"][2] == "k" and len(clean["why"]) == 3


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
    assert "47% below retail ($249.95 at Best Buy [R1])" in gap["sentence"]
    memo = Memo(price_low=82, price_high=88, headline="h", why=["k"], confidence="high", confidence_reason="c", watch_out=[])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["why"][1] == ("New is cheap: it's $250 at Best Buy [R1], and sealed new ones on eBay ask about half that (median $132), "
                               "so a used one much above $88 loses to new.")
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
    base = dict(price_low=82, price_high=88, headline="h", why=[], watch_out=[], confidence="high", confidence_reason="c")
    return Memo(**{**base, **kw})


# --- Key factors appear once ---


def test_why_bullets_are_deduped_against_codes_own():
    run = new_below_retail_run()
    gap = research.new_vs_retail(run)["sentence"]
    memo = memo_with(why=[gap, "Sealed new units on eBay sell well below the $249.95 retail price [R1].",
                                  "Free shipping is standard.", "Free shipping is standard!"])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert len(clean["why"]) == 3 and clean["why"][0].startswith("Comparable listings: 12 used listings mostly ask")
    assert clean["why"][1].startswith("New is cheap") and clean["why"][2] == "Free shipping is standard."


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
    assert "[E5]" not in clean["headline"] and "E1" in check["cited"] and not {"E5", "E6"} & set(check["cited"])


# --- Best Offer only when offers are common ---


def test_best_offer_is_a_factor_only_when_30_percent_of_comps_take_offers():
    run = run_with_sources(10)
    memo = memo_with(why=["Best Offer gives room to haggle", "Free shipping"])
    for sid in ("E1", "E2"):  # 2 of 10 take offers
        run.sources[sid]["buyingOptions"] = ["FIXED_PRICE", "BEST_OFFER"]
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["why"][1:] == ["Free shipping"]
    run.sources["E3"]["buyingOptions"] = ["FIXED_PRICE", "BEST_OFFER"]  # 3 of 10: 30%
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["why"][1:] == ["Best Offer gives room to haggle", "Free shipping"]
    assert research.offers_common({"comps": [{"best_offer": True}, {"best_offer": False}, {"best_offer": False}, {"best_offer": False}]}) is False


def test_retail_title_names_the_model_when_the_store_leaves_it_out():
    title, note = research.retail_title("Personalized AirPods Pro", "Apple AirPods Pro (2nd generation)")
    assert title == "AirPods Pro" and "personalization option" in note and "doesn't state the model" in note
    assert research.retail_title("Apple AirPods Pro 2 with USB-C", "Apple AirPods Pro (2nd generation)") == ("Apple AirPods Pro 2 with USB-C", "")
    assert research.retail_title("Apple iPhone 13 Pro 128GB", "Apple iPhone 13 Pro")[1] == ""


# --- Condition words -> eBay condition group ---

AF1 = Item(product="Nike Air Force 1 '07", variant="women's size 7", condition="like_new", condition_detail="new without tags",
           included="", search_query="Nike Air Force 1 07 womens size 7")


@pytest.mark.parametrize("words, intake, mapped", [
    ("new without tags", "like_new", "open_box"),
    ("new without box", "like_new", "open_box"),
    ("New without the box", "new", "open_box"),
    ("never worn", "like_new", "open_box"),
    ("unworn, tried on indoors", "used", "open_box"),
    ("NWOT", "new", "open_box"),
    ("worn once", "open_box", "like_new"),
    ("like new", "used", "like_new"),
    ("excellent condition", "used", "like_new"),
    ("like new, never worn", "like_new", "open_box"),  # unworn wins over "like new"
    ("new with tags", "like_new", "new"),
    ("good, light scratches", "used", "used"),
    ("sealed", "new", "new"),
    ("never used, for parts", "for_parts", "for_parts"),
])
def test_condition_words_map_to_the_ebay_group(words, intake, mapped):
    assert research.map_condition(AF1.model_copy(update={"condition": intake, "condition_detail": words})) == mapped


def test_condition_mapping_reads_the_sellers_message_too():
    item = AF1.model_copy(update={"condition_detail": ""})
    assert research.map_condition(item, "nike air force 1 07, size 7 for womens, new without tags") == "open_box"


def test_exact_comps_search_uses_the_mapped_condition_group(offline_market, monkeypatch):
    async def broken(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(research.Runner, "run", broken)
    run = Run(item=AF1.model_copy(update={"condition": "open_box"}), assumptions=[], trace=Trace())
    asyncio.run(research.run_researcher(run, brief(conditions=["like_new"]), 1))  # the Lead sent like_new
    search = next(c for c in run.trace.calls if c["name"].startswith("search_ebay"))
    assert search["args"]["conditions"] == ["open_box"]
    assert run.findings[0]["conditions"] == ["open_box"]


def test_open_box_is_ebays_new_without_box_or_tags():
    assert market.CONDITION_IDS["open_box"] == [1500]  # "New without box", "New without tags", "New other"


# --- Robust retail ceiling ---


def af1_retail_run(offers):
    run = Run(item=AF1.model_copy(update={"condition": "open_box"}), assumptions=[], trace=Trace())
    for i, price in enumerate([88.0, 90, 92, 94, 96, 98], 1):
        run.register("E", str(i), {"title": f"AF1 {i}", "price": price, "shippingCost": 0.0, "totalPrice": price, "condition": "New without box",
                                   "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": f"https://www.ebay.com/itm/{i}"})
    run.findings.append(fake_finding(n_kept=6))
    ids = []
    for i, (retailer, price, old) in enumerate(offers, 1):
        ids.append(run.register("R", str(i), {"title": "Nike Air Force 1 '07 Women's", "retailer": retailer, "price": f"${price}",
                                              "extracted_price": price, "old_price": f"${old}" if old else None, "extracted_old_price": old,
                                              "link": f"https://example.com/r{i}"}))
    f = fake_finding("retail", n_kept=0)
    f["kept_ids"], f["n_kept"] = ids, len(ids)
    run.findings.append(f)
    return run


def test_ceiling_is_the_major_retailer_median_not_one_cheap_deal():
    run = af1_retail_run([("Nike", 115.0, None), ("Foot Locker", 115.0, None), ("Dick's Sporting Goods", 110.0, None),
                          ("Kohl's", 51.0, 115.0), ("eBay - sneakerplug", 70.0, None), ("Shoe Palace", 120.0, None)])
    ref = research.retail_reference(run)
    assert ref["price"] == 115.0 and ref["id"] is None  # median of Nike, Foot Locker, Dick's (Kohl's $51 is an outlier)
    assert ref["outlier_ids"] == ["R4"] and set(ref["pool_ids"]) == {"R1", "R2", "R3"}
    assert ref["basis"].startswith("median of 3 major retailers") and ref["resale_to_retail_ratio"] == round(93.0 / 115.0, 2)
    assert ref["lowest_note"] == "One retailer, Kohl's, has it on sale for $51.00 (was $115.00) [R4]: a deal, not the ceiling."
    assert research.ceiling_text(ref).startswith("$115.00 (median of 3 major retailers: Dick")
    # a range above the $51 deal but under the robust ceiling doesn't warn
    clean, check = research.validate_memo(run, memo_with(price_low=90, price_high=100), research.confidence_cap(run))
    assert not any("ceiling" in w for w in check["warnings"])


def test_no_ceiling_without_a_mainstream_store_and_a_lone_offer_is_cited():
    # only independent / marketplace sellers: no ceiling at all, rather than their markup
    assert research.retail_reference(af1_retail_run([("Shoe Palace", 120.0, None), ("Hibbett", 110.0, None), ("Walmart - Seller", 99.0, None)])) is None
    lone = research.retail_reference(af1_retail_run([("Best Buy", 249.95, None)]))
    assert lone["id"] == "R1" and lone["price"] == 249.95 and lone["lowest_note"] is None
    assert research.ceiling_text(lone) == "$249.95 at Best Buy [R1]"


@pytest.mark.parametrize("query, expected", [
    ("Nike Air Force 1 07 womens size 7", "Nike Air Force 1 07 womens"),
    ("Nike Air Force 1 '07 sz 7.5 W", "Nike Air Force 1 '07"),
    ("Apple AirPods Pro 2", "Apple AirPods Pro 2"),
    ("iPhone 13 Pro 128GB", "iPhone 13 Pro 128GB"),
])
def test_retail_query_drops_shoe_sizes_only(query, expected):
    assert research.retail_query(query) == expected


def test_marketplace_sellers_on_a_big_store_are_not_the_store():
    assert research.is_major_retailer("Walmart", "Nike Air Force 1")
    assert not research.is_major_retailer("Walmart - Cellfeee", "Nike Air Force 1")
    assert research.is_major_retailer("Nike", "Nike Air Force 1")
    assert not research.is_major_retailer("Shoe Palace", "Nike Air Force 1")


def test_ceiling_counts_each_retailer_once_at_its_lowest_price():
    # one store's ten colorways don't outvote another store
    offers = [("Foot Locker", 125.0, None)] * 5 + [("Foot Locker", 99.0, None)] * 3 + [("Champs Sports", 90.0, None)]
    ref = research.retail_reference(af1_retail_run(offers))
    assert ref["price"] == 94.5 and len(ref["pool_ids"]) == 2 and ref["basis"].startswith("median of 2 major retailers")


def test_a_tag_before_its_price_is_checked_against_that_price():
    run = run_with_sources(12)  # E1 = $81 ... E12 = $92
    text = "listings like [E1] at $81 and [E5] at $85."
    assert research.fix_stat_citations(text, run) == text  # both right: untouched, no stat label on $81
    assert research.fix_stat_citations("like [E1] at $81 and [E9] at $85.", run) == "like [E1] at $81 and at $85."


def test_a_basis_named_after_the_tag_is_not_labelled_twice():
    run = af1_retail_run([("Nike", 115.0, None), ("Foot Locker", 115.0, None), ("Dick's Sporting Goods", 110.0, None)])
    assert research.fix_stat_citations("below the retail ceiling of $115.00 [R3] (median of 3 major retailers).", run) == \
        "below the retail ceiling of $115.00 (median of 3 major retailers)."
    assert research.fix_stat_citations("below the retail ceiling of $115.00 [R3].", run).startswith(
        "below the retail ceiling of $115.00 (median of 3 major retailers: Dick")


# --- Fixes from the live scenario tests ---


@pytest.mark.parametrize("words, intake, mapped", [
    ("cracked back but works", "for_parts", "used"),  # damaged but working is used
    ("cracked screen, still works fine", "for_parts", "used"),
    ("doesn't turn on", "used", "for_parts"),
    ("won't charge", "used", "for_parts"),
    ("not working", "used", "for_parts"),
    ("dead battery, no power", "used", "for_parts"),
    ("good, light scratches", "used", "used"),
])
def test_working_vs_not_working(words, intake, mapped):
    assert research.map_condition(ITEM.model_copy(update={"condition": intake, "condition_detail": words})) == mapped


MACBOOK = "MacBook Air M2, 8GB RAM, 256GB"


@pytest.mark.parametrize("title, reason", [
    ("Apple MacBook Air M2 8GB 256GB Midnight", None),  # bare 8GB is the RAM, not a second storage size
    ("Apple MacBook Air 13.6\" M2 8GB RAM 256GB SSD", None),
    ("MacBook Air M2 16GB 256GB", "different RAM: 16gb"),
    ("MacBook Air M2 8GB RAM 512GB SSD", "different storage: 512gb"),
    ("MacBook Air M1 8GB 256GB", "different chip: M1"),
    ("MacBook Air M2 8GB", "variant not stated: 256gb"),
])
def test_laptop_chip_ram_and_storage(title, reason):
    assert research.laptop_variant_exclusion(title, MACBOOK) == reason


@pytest.mark.parametrize("title, reason", [
    ("Sony PS5 Slim Disc Edition 1TB Console", None),
    ("PS5 Slim Disc 1TB with 1 controller", None),
    ("R U GTA6 Ready? Sony PS5 Slim Disc (1TB CFI-2015) Bundle.", "bundle with extras"),
    ("PS5 Slim Disc Edition 1TB w 2 controllers", "bundle with extras"),
    ("PS5 Slim Disc + 3 Games", "bundle with extras"),
    ("PS5 Slim Digital Edition 1TB", "different edition: digital"),
])
def test_console_bundles_and_editions(title, reason):
    assert research.console_exclusion(title, "PS5 Slim disc edition with 1 controller") == reason


def test_categories_for_laptops_and_consoles():
    def cat(product):
        return research.item_profile(ITEM.model_copy(update={"product": product, "search_query": product}))["category"]
    assert cat("Apple MacBook Air M2") == "laptop" and cat("Sony PS5 Slim") == "console"
    assert cat("Sony PS5 DualSense controller") == "console" and cat("Nintendo Switch OLED") == "console"


def test_one_of_a_kind_items_are_priced_from_similar_items_at_low_confidence():
    quilt = ITEM.model_copy(update={"product": "Vintage handmade quilt", "variant": "queen", "condition_detail": "good, from the 1970s",
                                    "search_query": "vintage handmade quilt queen"})
    assert research.item_profile(quilt)["similar_items"] is True
    assert research.item_profile(CHANEL)["similar_items"] is False  # "vintage Chanel" is still a model with comps
    run = run_with_sources(12)
    run.item = quilt
    assert research.confidence_cap(run) == ("low", "the comps are similar items, not exact matches")
    clean, _ = research.validate_memo(run, memo_with(why=["Quilts ask about $90."]), research.confidence_cap(run))
    assert clean["watch_out"][0].startswith("Similar items, not exact matches")
    assert clean["confidence"] == "low"


@pytest.mark.parametrize("text, weapon", [
    ("selling a hunting rifle", True), ("Glock 19 pistol", True), ("box of 9mm ammo", True), ("AR-15 lower", True),
    ("Ryobi glue gun", False), ("Theragun massage gun", False), ("Nerf gun lot", False), ("Guns N' Roses vinyl", False),
    ("AirPods Pro 2", False),
])
def test_weapons_are_recognised(text, weapon):
    assert research.is_weapon(text) is weapon


def test_a_failed_retail_search_is_reported_as_failed_not_as_not_found():
    run = new_run()
    failed = fake_finding("retail", n_kept=0)
    failed["errors"] = {"retail": "RuntimeError: SerpApi budget reached"}
    run.findings.append(failed)
    status = research.retail_status(run)
    assert status["status"] == "search_failed" and "couldn't be checked" in status["note"]


def test_weapon_is_declined_before_any_model_call(monkeypatch):
    async def must_not_run(*a, **k):
        raise AssertionError("no model call for a weapon")

    monkeypatch.setattr(research.Runner, "run", must_not_run)
    out = asyncio.run(research.answer([{"role": "user", "content": "selling a hunting rifle"}], 0))
    assert out["memo"] is None and not out["clarifying"] and "can't help price firearms" in out["response"]
    assert [c["name"] for c in out["tool_calls"]] == ["decline"]


# --- Quota and credential errors say what happened ---


@pytest.mark.parametrize("service, status, text, kind", [
    ("SerpApi", None, "Your account has run out of searches.", "quota"),
    ("SerpApi", 429, "", "quota"),
    ("SerpApi", 401, "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key", "auth"),
    ("eBay", 401, "invalid_client", "auth"),
    ("eBay", 500, "Internal error", "other"),
])
def test_api_errors_are_classified(service, status, text, kind):
    e = market.classify(service, status, text)
    assert e.kind == kind and e.service == service
    if kind == "auth":
        assert ("SERPAPI_KEY" if service == "SerpApi" else "EBAY_CLIENT_ID") in str(e)


def test_quota_error_reaches_the_memo_and_isnt_retried():
    run = new_run()
    f = fake_finding("retail", n_kept=0)
    f["errors"] = {"retail": "ApiError: " + str(market.classify("SerpApi", None, "Your account has run out of searches."))}
    run.findings.append(f)
    status = research.retail_status(run)
    assert status["status"] == "search_failed" and "SerpApi quota used up" in status["note"]
    gap = research.find_gaps([fake_finding(), f])[0]
    assert "Retrying won't help" in gap and not gap.startswith("retail search failed (retail): ApiError")


def test_ebay_failure_is_reported_as_a_failure_not_as_no_listings():
    run = new_run()
    f = fake_finding(n_kept=0)
    f["n_found"], f["errors"] = 0, {"used": "ApiError: " + str(market.classify("eBay", 401, "invalid_client"))}
    run.findings.append(f)
    response, memo = asyncio.run(research.write_memo(run))
    assert memo is None and response.startswith("I couldn't search eBay") and "EBAY_CLIENT_ID" in response


MBA = Item(product="Apple MacBook Air M2", variant="8GB RAM, 256GB", condition="used", condition_detail="good", included="",
           search_query="MacBook Air M2 8GB 256GB")
PS5 = Item(product="Sony PS5 Slim", variant="disc edition", condition="used", condition_detail="with 1 controller", included="1 controller",
           search_query="PS5 Slim disc")


@pytest.mark.parametrize("title, item, successor, reason", [
    ("2022 Apple M2 MacBook Air 13\" 8GB/256GB SSD (Midnight)", MBA, False, None),
    ("13-inch MacBook Air (M5) - Midnight 10-core CPU / 24GB / 2TB", MBA, True, "different RAM: 24gb"),  # successor: chip may differ, specs may not
    ("13-inch Apple MacBook Air M5", MBA, False, "different chip: M5"),
    ("Apple MacBook Air M2 chip", MBA, False, None),  # a store title that leaves the specs out is kept
    ("Apple MacBook Air M2 8GB 256GB - Refurbished", MBA, False, "not new: refurbished / used offer"),
    ("Sony PlayStation 5 Slim 1TB Disc Bundle Console", PS5, False, "bundle with extras"),
    ("Sony PS5 slim disc edition w/2 controller, charging station", PS5, False, "bundle with extras"),
    ("Sony PlayStation 5 Slim Digital Edition", PS5, False, "different edition: digital"),
    ("Sony PlayStation 5 Slim Disc Console", PS5, False, None),
])
def test_store_offers_get_the_same_code_checks(title, item, successor, reason):
    assert research.retail_title_exclusion(title, item, successor) == reason


def test_a_guessed_storage_is_left_out_of_the_variant():
    guessed = PS5.model_copy(update={"variant": "disc edition, 1TB", "search_query": "PS5 Slim Disc 1TB"})
    fixed = research.drop_guessed_variant(guessed, "PS5 Slim disc edition, used, with 1 controller")
    assert (fixed.variant, fixed.search_query) == ("disc edition", "PS5 Slim Disc")
    said = research.drop_guessed_variant(guessed, "PS5 Slim disc 1 TB, used")  # the seller said it
    assert said.variant == "disc edition, 1TB"
    phone = ITEM.model_copy(update={"variant": "128GB, unlocked", "search_query": "iPhone 13 Pro 128GB unlocked"})
    assert research.drop_guessed_variant(phone, "iPhone 13 Pro 128GB, used").variant == "128GB"


@pytest.mark.parametrize("title, item_text, reason", [
    ("Sony PlayStation 5 Disc Console", "Sony PS5 Slim disc edition", "different model: not Slim"),
    ("Sony PS5 Slim Disc Console", "Sony PS5 disc edition", "different model: Slim"),
    ("Nintendo Switch Neon 32GB", "Nintendo Switch OLED", "different model: not OLED"),
    ("Nintendo Switch OLED White", "Nintendo Switch OLED", None),
])
def test_console_revisions(title, item_text, reason):
    assert research.console_exclusion(title, item_text) == reason


def test_citations_to_excluded_listings_are_removed():
    run = run_with_sources(12)
    run.register("E", "x", {"title": "bundle", "price": 700.0, "shippingCost": 0.0, "totalPrice": 700.0, "condition": "Used",
                            "buyingOptions": ["FIXED_PRICE"], "itemWebUrl": "https://www.ebay.com/itm/x"})  # E13
    run.findings[0]["excluded"] = [{"id": "E13", "title": "bundle", "reason": "bundle with extras", "by": "code"}]
    clean, check = research.validate_memo(run, memo_with(why=["Bundles ask $700 [E13]; singles $81 [E1]."]), research.confidence_cap(run))
    assert not any("[E13]" in w for w in clean["why"]) and any("$81 [E1]" in w for w in clean["why"])
    assert "E13" not in check["cited"] and any("removed citations to excluded listings: E13" in w for w in check["warnings"])


# --- (b) third-party sellers never set the ceiling ---


def test_third_party_sellers_are_dropped_in_code_and_mean_no_mainstream_price(monkeypatch):
    offer = {"title": "Sony PlayStation 5 Slim Disc Console", "price": "$799", "extracted_price": 799.0, "old_price": None,
             "extracted_old_price": None, "second_hand_condition": None, "is_resale": False}
    offers = [{**offer, "retailer": r, "link": f"https://s/{i}", "product_id": str(i)}
              for i, r in enumerate(["Tech For Less", "The Device Depot", "Walmart - Seller", "uShopMall.com"])]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": offers})
    run = Run(item=PS5, assumptions=[], trace=Trace())
    state = ResearcherState(run, brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "PS5 Slim"))
    assert not state.shown and all(d["reason"].startswith(research.THIRD_PARTY) for d in state.dropped)  # decided in code
    research.do_review(state, [])
    run.findings.append(state.finding)
    assert state.finding["no_mainstream"] and research.find_gaps(run.findings, PS5) == []  # final: no retry, no successor
    assert research.retail_reference(run) is None
    status = research.retail_status(run)
    assert status["status"] == "no_mainstream" and status["note"].startswith("No mainstream retail price")


@pytest.mark.parametrize("retailer, major", [
    ("Best Buy", True), ("Walmart", True), ("Walmart - Seller", False), ("PlayStation Direct", True), ("Dyson Official", True),
    ("Gap", True), ("Singapore Gap Outlet - Seller", False), ("Tech For Less", False), ("farfetch.com", False),
])
def test_mainstream_retailers(retailer, major):
    assert research.is_major_retailer(retailer, "Dyson V8") is major


# --- (a) used above retail on an everyday item ---


def test_above_retail_is_flagged_only_for_everyday_used_items():
    run = run_with_sources(12, retail_price=60.0)  # comps ~$86 vs $60 retail
    assert research.above_retail(run)
    for item in (CHANEL, ITEM.model_copy(update={"product": "Vintage handmade quilt", "search_query": "vintage handmade quilt"}),
                 ITEM.model_copy(update={"product": "AirPods Pro 2 limited edition"}), ITEM.model_copy(update={"condition": "new"})):
        run.item = item
        assert not research.above_retail(run), item.product


# --- (b2) appliance model matching ---


@pytest.mark.parametrize("title, product, reason", [
    ("Dyson V10 Animal Cordless Vacuum", "Dyson V8", "different model: V10"),
    ("Dyson V8 Animal SV10 Cordless Vacuum", "Dyson V8", None),  # SV10 is the V8's own model number
    ("Dyson V8 V10 V11 compatible brush", "Dyson V8", None),  # names the V8 too: left to the review
    ("Dyson V8 Absolute Cordless", "Dyson V8 Animal", "different model line: absolute"),
    ("Dyson V8 Animal+ Cordless", "Dyson V8 Animal", None),
    ("Samsung Galaxy S24 128GB", "Samsung Galaxy S23", "different model: S24"),
    ("Sony PS4 Slim 1TB", "Sony PS5 Slim", "different model: PS4"),
])
def test_model_codes_and_product_lines(title, product, reason):
    assert research.model_code_exclusion(title, product) == reason


# --- (c) for parts ---


def test_for_parts_pricing_is_capped_at_medium_and_skips_working_phones(monkeypatch):
    run = run_with_sources(12)
    run.item = ITEM.model_copy(update={"condition": "for_parts", "condition_detail": "doesn't turn on"})
    assert research.confidence_cap(run)[0] == "medium"
    titles = ["Apple iPhone 12 128GB unlocked - does not turn on", "Apple iPhone 12 128GB unlocked cracked screen works fine",
              "Apple iPhone 12 128GB unlocked for parts"]
    records = [{"itemId": str(i), "title": t, "price": 120.0, "currency": "USD", "shippingCost": 0.0, "totalPrice": 120.0,
                "condition": "For parts or not working", "conditionId": "7000", "buyingOptions": ["FIXED_PRICE"],
                "itemCreationDate": None, "itemWebUrl": f"https://www.ebay.com/itm/{i}", "image": None} for i, t in enumerate(titles)]
    monkeypatch.setattr(market, "ebay_search", lambda query, condition=None, limit=50, **kw: {"query": query, "filter": "", "total": 3, "items": records})
    phone = Item(product="Apple iPhone 12", variant="128GB unlocked", condition="for_parts", condition_detail="doesn't turn on", included="",
                 search_query="iPhone 12 128GB unlocked")
    state = ResearcherState(Run(item=phone, assumptions=[], trace=Trace()), brief(conditions=["for_parts"]), 1)
    asyncio.run(research.do_search_ebay(state, "iPhone 12 128GB unlocked", ["for_parts"]))
    assert [d["reason"] for d in state.dropped] == ["works (yours doesn't): not comparable for parts"] and len(state.shown) == 2


# --- (d) comp counts ---


def test_code_sentences_with_other_counts_are_not_count_mismatches():
    run = new_below_retail_run()  # code adds "... across 4 sealed new listings ..."
    clean, check = research.validate_memo(run, memo_with(headline="List at $82-$88 from 12 kept comps"), research.confidence_cap(run))
    assert clean["why"][1].startswith("New is cheap: it's $250") and not any("comp counts" in w for w in check["warnings"])


def test_a_count_still_wrong_after_the_retry_is_fixed_in_code(monkeypatch):
    run = run_with_sources(12)
    bad = Memo(price_low=82, price_high=88, headline="List at $82-$88 from 44 comparable listings", confidence="high",
               confidence_reason="Based on 44 comps.", watch_out=[],
               why=["44 comps agree", "Free shipping is common", "The 18 listings with free shipping ask more."])

    class Result:
        final_output = bad

    async def fake_run(agent, prompt, **kw):
        return Result()

    monkeypatch.setattr(research.Runner, "run", fake_run)
    _, memo = asyncio.run(research.write_memo(run))
    assert "12 comparable listings" in memo["headline"] and memo["confidence_reason"].startswith("Based on 12 comps")
    assert memo["why"][1:] == ["Free shipping is common"]  # the two bullets with wrong counts are gone
    assert not any("comp counts" in w for w in memo["citation_check"]["warnings"])
    assert any(c["name"] == "recount (code fix)" for c in run.trace.calls)


# --- (e) apparel size and color ---

LULU = ("size 6, black", "Lululemon Align leggings")
AF1_W7 = ("women's size 7", "Nike Air Force 1 '07")


@pytest.mark.parametrize("title, item, reason", [
    ('lululemon Align Pant 25" Size 6 Black', LULU, None),
    ('lululemon Align High-Rise Pant 28" Size 4 Black', LULU, "different size: 4"),
    ("Lululemon Align Leggings Sz 6 Navy", LULU, "different color: navy"),
    ("Lululemon Align Leggings Black/White Size 6", LULU, None),
    ("Lululemon Align Leggings Heathered Black", LULU, None),  # size not stated: left to the review
    ("Nike Air Force 1 '07 Womens Size 7 White", AF1_W7, None),
    ("Nike Air Force 1 '07 Women's Shoes White", AF1_W7, None),  # the 1 is not a size
    ("Nike Air Force 1 '07 Womens 8.5 White", AF1_W7, "different size: 8.5"),
    ("Nike Air Force 1 07 7W Triple White", AF1_W7, None),
])
def test_apparel_size_and_color(title, item, reason):
    assert research.apparel_variant_exclusion(title, *item) == reason


def test_apparel_category():
    def cat(product):
        return research.item_profile(ITEM.model_copy(update={"product": product, "search_query": product}))["category"]
    assert cat("Lululemon Align leggings") == "apparel" and cat("Nike Air Force 1 '07") == "apparel"


# --- Retail review can't overrule code on seller type, price level or condition ---


def test_reviewer_cannot_drop_a_mainstream_offer_as_inflated_or_preowned(monkeypatch):
    offer = {"price": "$799.95", "extracted_price": 799.95, "old_price": None, "extracted_old_price": None, "second_hand_condition": None, "is_resale": False}
    offers = [{**offer, "title": "Sony PlayStation 5 Slim Disc Console", "retailer": "Walmart", "link": "https://s/1", "product_id": "1"},
              {**offer, "title": "Sony PlayStation 5 Slim Console", "retailer": "GameStop", "extracted_price": 649.99, "link": "https://s/2", "product_id": "2"},
              {**offer, "title": "Sony PlayStation 5 Slim Disc Console + Gran Turismo", "retailer": "Target", "link": "https://s/3", "product_id": "3"}]
    monkeypatch.setattr(market, "retail_search", lambda query, **kw: {"query": query, "items": offers})
    run = Run(item=PS5, assumptions=[], trace=Trace())
    state = ResearcherState(run, brief("retail", ()), 1)
    asyncio.run(research.do_search_retail(state, "PS5 Slim Disc"))
    out = research.do_review(state, [Exclusion(id="R1", reason="inflated third-party seller price via Walmart marketplace"),
                                     Exclusion(id="R2", reason="pre-owned from GameStop"),
                                     Exclusion(id="R3", reason="bundle with a game")])
    assert [o["id"] for o in out["overruled_by_code"]] == ["R1", "R2"]
    assert state.finding["kept_ids"] == ["R1", "R2"]  # the bundle exclusion is about the product: it stands
    run.findings.append(state.finding)
    ref = research.retail_reference(run)
    assert ref["price"] == round((799.95 + 649.99) / 2, 2) and ref["basis"].startswith("median of 2 major retailers")


def test_run_together_bundle_titles_are_bundles():
    assert research.console_exclusion("PS5 Slim Disc 1TB Console with Extra ControllerBundle", "PS5 Slim disc with 1 controller") == "bundle with extras"
    assert research.console_exclusion("Sony PS5 Slim Console with Extra DualSense", "PS5 Slim disc with 1 controller") == "bundle with extras"


# --- The short memo: one takeaway, up to 3 Why, 1-3 item-specific Watch out, each fact once ---


def test_takeaway_with_jargon_or_too_long_is_replaced_by_codes_plain_one():
    run = run_with_sources(12)
    for bad in ("List at $82-$88: between the Q1 of $84 and the median $86.50 (price + shipping).",
                " ".join(["word"] * 30)):
        clean, check = research.validate_memo(run, memo_with(headline=bad), research.confidence_cap(run))
        assert clean["headline"] == "List at $82–88 with free shipping, around what most similar used listings ask right now."  # $88 > the $86.50 median
        assert any("headline rewritten" in w for w in check["warnings"])
    clean, _ = research.validate_memo(run, memo_with(headline="List at $82-88 with free shipping. Most pairs ask more."), research.confidence_cap(run))
    assert clean["headline"] == "List at $82-88 with free shipping."  # one sentence


def test_why_has_at_most_three_different_facts_and_code_writes_the_comps_one():
    run = run_with_sources(12)
    memo = memo_with(why=["Median $86.50 across 12 comps [E5].", "Condition: light wear on the case.", "Shipping: most ship free, like $81 [E1].",
                          "Buyers negotiate: room for offers.", "Condition: light wear on the case."])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert len(clean["why"]) == 3
    assert clean["why"][0].startswith("Comparable listings: 12 used listings mostly ask $84 [E4] to $89 [E9], with a median of $86")
    assert "Median $86.50 across 12 comps [E5]." not in clean["why"]  # comps are code's bullet
    assert clean["why"][1:] == ["Condition: light wear on the case.", "Shipping: most ship free, like $81 [E1]."]


def test_watch_out_keeps_only_item_specific_points():
    run = run_with_sources(12)
    run.assumptions = ["Port version not given: comps include both Lightning and USB-C."]
    memo = memo_with(watch_out=["Asking prices are not sold prices, and active listings skew high.",
                                "Port version not given: comps include both Lightning and USB-C.",
                                "Retail price couldn't be verified online.",
                                "8 listings were excluded as accessories.",
                                "USB-C cases ask a little more than Lightning: if yours is USB-C, aim for the top of the range."])
    clean, _ = research.validate_memo(run, memo, research.confidence_cap(run))
    assert clean["watch_out"] == ["USB-C cases ask a little more than Lightning: if yours is USB-C, aim for the top of the range."]


def test_confidence_reason_is_one_short_sentence():
    run = run_with_sources(12)
    long = "Twelve kept comps sit close together. " + " ".join(["more"] * 40)
    clean, _ = research.validate_memo(run, memo_with(confidence_reason=long), research.confidence_cap(run))
    assert clean["confidence_reason"] == "Twelve kept comps sit close together."


def test_rendered_memo_says_each_thing_once():
    run = run_with_sources(20)
    run.assumptions = ["Port version not given: comps include both Lightning and USB-C."]
    run.findings[0]["excluded"] = [{"id": "E90", "title": "x", "reason": "case only", "by": "code"}]
    clean, _ = research.validate_memo(run, memo_with(why=["Buyers negotiate: room for offers."]), research.confidence_cap(run))
    markdown, _ = research.render(run, clean, {"comps": research.comp_rows(run), "tiers": [], "retail": research.retail_rows(run)}, [])
    memo_part = markdown.split("### Comparable listings")[0]
    assert "### Why" in memo_part and "Limitations" not in markdown and "Key factors" not in markdown
    assert markdown.count("not a prediction of what it will sell for") == 1
    assert "Excluded" not in memo_part and markdown.count("Excluded 1 of 21 listings") == 1  # only with the comps
    assert "*Assumed: Port version not given: comps include both Lightning and USB-C.*" in memo_part
    assert "middle half" in markdown.split("### Comparable listings")[1] and "middle half" not in memo_part
    assert research.assumed_line(["a", "b", "c"]).startswith("*Assumed:*\n- *a*")


def test_resale_only_retry_names_the_sellers_plain_product_name():
    switch = ITEM.model_copy(update={"product": "Nintendo Switch OLED", "search_query": "Nintendo Switch OLED"})
    f = fake_finding("retail", n_kept=0, only_resale=True)
    f["queries"] = ["Nintendo Switch OLED Console"]
    [gap] = research.find_gaps([fake_finding(), f], switch)
    assert "Try once more with exactly 'Nintendo Switch OLED'" in gap
    f["queries"] = ["Nintendo Switch OLED"]  # the plain name already tried: the generic advice
    assert "full product name a store would use" in research.find_gaps([fake_finding(), f], switch)[0]
