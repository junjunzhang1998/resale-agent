"""Live scenario tests: real eBay, SerpApi and Gemini calls through the research pipeline.

    uv run python scripts/scenario_tests.py                  # every scenario
    uv run python scripts/scenario_tests.py --only rifle,quilt
    uv run python scripts/scenario_tests.py --serpapi-budget 15

Each scenario runs like a /chat session (multi-turn ones share one session's turns). Full
responses go to scenario_results/<name>.json; scenario_results/SUMMARY.md is rebuilt from every
saved result, so a rerun of a few scenarios keeps the others' rows.

SerpApi has a small free quota: the disk cache is on (48h unless CACHE_TTL_HOURS is set), and live
SerpApi calls are counted and capped by --serpapi-budget. Past the cap a retail search fails, which
shows up in that scenario's notes.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CACHE_TTL_HOURS", "48")

import market  # noqa: E402
import research  # noqa: E402

OUT = ROOT / "scenario_results"

# --- Helpers the checks use ---


def exact_conditions(r: dict) -> list[str] | None:
    """The condition group the first exact_comps search used."""
    for c in r["tool_calls"]:
        if c["agent"] == "researcher:exact_comps" and c["name"].startswith("search_ebay") and isinstance(c["args"], dict):
            return c["args"].get("conditions")
    return None


def researched(r: dict) -> bool:
    return any(c["name"].startswith(("dispatch_researchers", "search_")) for c in r["tool_calls"])


def kept_titles(r: dict) -> list[str]:
    return [c["title"] for c in (r["memo"] or {}).get("comps", [])]


def memo_text(r: dict) -> str:
    m = r["memo"] or {}
    return " ".join([m.get("headline", ""), m.get("rationale", ""), m.get("confidence_reason", ""),
                     *m.get("key_factors", []), *m.get("limitations", [])])


def check(name: str, ok: bool, detail: str = "") -> dict:
    return {"check": name, "pass": bool(ok), "detail": detail}


def condition_is(group: str):
    def run(r):
        got = exact_conditions(r)
        return [check(f"exact_comps searched {group}", got == [group], f"searched {got}")]
    return run


def has_memo(r):
    return check("priced (memo returned)", r["memo"] is not None, "" if r["memo"] else r["response"][:120])


def ceiling_mainstream(r):
    """The ceiling isn't one far-below offer and no refurbished/used offer is in its pool."""
    m = r["memo"] or {}
    ref = m.get("retail_reference")
    if not ref:
        return [check("retail ceiling found", False, (m.get("retail_status") or {}).get("note", ""))]
    sources = m.get("sources", {})
    pool_titles = [(sources.get(i) or {}).get("title", "") for i in ref.get("pool_ids", [])]
    refurb = [t for t in pool_titles if re.search(r"refurb|renewed|pre-owned|used|open box", t or "", re.I)]
    out = [check("retail ceiling found", True, f"{research._money(ref['price'])} ({ref.get('basis')})")]
    out.append(check("ceiling not a far-below outlier", ref["price"] >= 0.6 * (ref.get("median_new") or ref["price"]),
                     f"ceiling {ref['price']} vs median of all offers {ref.get('median_new')}"))
    out.append(check("no refurbished/used offer in the ceiling pool", not refurb, "; ".join(refurb)[:160]))
    return out


def no_titles(pattern: str, label: str):
    def run(r):
        bad = [t for t in kept_titles(r) if re.search(pattern, t, re.I)]
        return [check(label, r["memo"] is not None and not bad, "; ".join(bad[:3])[:200])]
    return run


def all_titles(pattern: str, label: str):
    def run(r):
        bad = [t for t in kept_titles(r) if not re.search(pattern, t, re.I)]
        return [check(label, r["memo"] is not None and not bad, "; ".join(bad[:3])[:200])]
    return run


def no_count_warning(r):
    w = [x for x in ((r["memo"] or {}).get("citation_check") or {}).get("warnings", []) if "comp counts" in x]
    return check("no comp-count warning", r["memo"] is not None and not w, "; ".join(w))


def ceiling_mainstream_or_none(r):
    """(b): a ceiling only from big-box/brand stores; otherwise 'no mainstream retail price'."""
    m = r["memo"] or {}
    ref, status = m.get("retail_reference"), (m.get("retail_status") or {}).get("status")
    if ref:
        names = {x["id"]: x["retailer"] for x in m.get("retail", [])} | {i: s.get("detail", "") for i, s in m.get("sources", {}).items()}
        ok = all(research.is_major_retailer(names.get(i, ""), m["item"]["product"]) for i in ref.get("pool_ids", []))
        return check("ceiling only from mainstream stores", ok, f"{research._money(ref['price'])} ({ref.get('basis')})")
    return check("ceiling only from mainstream stores (none: says no mainstream price)", status == "no_mainstream", str(status))


# --- Scenarios: name -> (turns, checks) ---

COLORS = r"\b(navy|blue|grey|gray|green|red|pink|white|brown|beige|purple|olive|tan|camo|burgundy|maroon|teal|orange|yellow|nomad|heathered)\b"

SCENARIOS = {
    # condition wording
    "cond_sealed_controller": (["brand new sealed PS5 DualSense controller, white"], [has_memo, condition_is("new")]),
    "cond_af1_worn_once": (["Nike Air Force 1 07 womens size 7, worn once, basically perfect"], [has_memo, condition_is("like_new"), no_count_warning]),
    "cond_switch_open_box": (["Nintendo Switch OLED, open box, never used"], [has_memo, condition_is("open_box")]),
    "cond_iphone_cracked": (["iPhone 12 128GB unlocked, cracked back but works"], [has_memo, condition_is("used"),
        lambda r: [check("lower sub-grade when graded", not (r["memo"] or {}).get("grade") or r["memo"]["grade"]["grade"] in ("Fair", "Poor"),
                         str(((r["memo"] or {}).get("grade") or {}).get("grade")))]]),
    "cond_iphone_dead": (["iPhone 12 128GB unlocked, doesn't turn on"], [has_memo, condition_is("for_parts"),
        lambda r: [check("kept comps are listed for parts", r["memo"] is not None and all("parts" in (c["condition"] or "").lower() for c in r["memo"]["comps"]),
                         "; ".join(sorted({c["condition"] for c in (r["memo"] or {}).get("comps", [])})))],
        no_titles(r"(works|working|functional)(?!.*(not|doesn'?t|won'?t))", "no kept comp says it works"),
        lambda r: [check("confidence capped at medium", (r["memo"] or {}).get("confidence") in ("low", "medium"), str((r["memo"] or {}).get("confidence")))]]),
    "cond_af1_nwot": (["nike air force 1 07, size 7 for womens, new without tags"], [has_memo, condition_is("open_box")]),
    # retail ceiling
    "retail_switch_oled": (["Nintendo Switch OLED, used, good condition"], [has_memo, ceiling_mainstream]),
    "retail_dyson_v8": (["Dyson V8 cordless vacuum, used, good condition"], [has_memo, ceiling_mainstream, no_count_warning,
        no_titles(r"V(7|10|11|12|15)|Gen5", "no V7/V10/V11/V12/V15 or Gen5 in kept comps"),
        lambda r: [check("no new units in kept comps", r["memo"] is not None and not any(re.search(r"new", c["condition"] or "", re.I) for c in r["memo"]["comps"]))]]),
    # variants
    "variant_ps5_slim": (["PS5 Slim disc edition, used, with 1 controller"], [has_memo, ceiling_mainstream_or_none,
        no_titles(r"\bbundle\b|\b(2|3|4|two|three|four)\s*(x\s*)?controllers?\b|\b(with|w/|\+|plus|&)\s*(\d+\s+)?games?\b|\bgames? included\b|\bdigital\b",
                  "no bundles with extra games/controllers, no digital edition")]),
    "variant_lululemon": (["Lululemon Align leggings size 6, black, used"], [has_memo,
        no_titles(r"\b(size|sz)\.?\s*(?!6\b)\d{1,2}\b", "no kept comp states another size"),
        no_titles(COLORS, "no kept comp names a non-black color")]),
    "variant_macbook": (["MacBook Air M2 8GB RAM 256GB, used, good"], [has_memo,
        all_titles(r"\bM2\b", "every kept comp is M2"),
        no_titles(r"\bM[134]\b|\b(16|24|32)\s*GB\b|\b(128|512)\s*GB\b|\b[12]\s*TB\b", "no other chip, RAM or storage stated")]),
    # clarification
    "clarify_macbook": (["I want to sell my MacBook", "it's an Air"], []),  # checked by turn below
    "clarify_misspelled_airpods": (["airpod pro 2nd gen gud condishun"], [has_memo,
        lambda r: [check("understood as AirPods Pro 2", bool(re.search(r"airpods? pro", (r["memo"] or {}).get("item", {}).get("product", ""), re.I)),
                         (r["memo"] or {}).get("item", {}).get("product", ""))]]),
    "clarify_two_items": (["I want to sell my AirPods and my iPhone"], [
        lambda r: [check("asks which item (or prices one and says so)",
                         (r["clarifying"] and bool(re.search(r"which|one at a time|first|both", r["response"], re.I)))
                         or (r["memo"] is not None and bool(re.search(r"iphone|airpods", " ".join((r["memo"] or {}).get("assumptions", [])), re.I))),
                         r["response"][:160])]]),
    # similar items
    "similar_quilt": (["vintage handmade quilt, queen size, from the 1970s, good condition"], [has_memo,
        lambda r: [check("says comps are similar items, not exact matches", bool(re.search(r"similar", memo_text(r), re.I))),
                   check("lower confidence (not high)", (r["memo"] or {}).get("confidence") in ("low", "medium"), str((r["memo"] or {}).get("confidence")))]]),
    # out of scope / sensitive
    "scope_tutoring": (["how much should I charge for tutoring?"], [
        lambda r: [check("scope reply, no research", r["memo"] is None and not researched(r), r["response"][:160])]]),
    "sensitive_rifle": (["selling a hunting rifle"], [
        lambda r: [check("polite decline, no research", r["memo"] is None and not researched(r)
                         and bool(re.search(r"can't|cannot|can not|not able|unable|don't|do not|won't", r["response"], re.I)), r["response"][:160])]]),
}


def clarify_macbook_checks(results: list[dict]) -> list[dict]:
    first, second = results
    q2 = second["response"]
    return [
        check("turn 1 asks a question", first["clarifying"] and first["memo"] is None, first["response"][:120]),
        check("turn 2 asks again, no memo", second["clarifying"] and second["memo"] is None, q2[:160]),
        check("turn 2 asks for chip/RAM/storage", bool(re.search(r"chip|processor|M1|M2|M3", q2, re.I))
              and bool(re.search(r"RAM|memory", q2, re.I)) and bool(re.search(r"storage|SSD|GB", q2, re.I)), q2[:160]),
    ]


# --- SerpApi budget ---

serpapi_live = {"n": 0, "budget": 15}


def guard_serpapi():
    original = market._cached

    def counted(kind, params, fetch, use_cache=True):
        if kind != "serpapi":
            return original(kind, params, fetch, use_cache)

        def budgeted():
            if serpapi_live["n"] >= serpapi_live["budget"]:
                raise market.ApiError("SerpApi", "budget", "SerpApi budget for this test run reached (scripts/scenario_tests.py --serpapi-budget).")
            serpapi_live["n"] += 1
            return fetch()
        return original(kind, params, budgeted, use_cache)

    market._cached = counted


# --- Running ---


async def run_session(messages: list[str]) -> list[dict]:
    """Like app.py's /chat: one session, the turns and clarification count carried over."""
    turns, clarifications, results = [], 0, []
    for text in messages:
        turns.append({"role": "user", "content": text})
        t = time.time()
        try:
            r = await asyncio.wait_for(research.answer(turns, clarifications), 240)
        except Exception as e:
            r = {"response": f"Research failed: {type(e).__name__}: {str(e)[:300]}", "tool_calls": [], "memo": None, "clarifying": False}
        r["seconds"] = round(time.time() - t, 1)
        r["input"] = text
        results.append(r)
        clarifications = clarifications + 1 if r["clarifying"] else 0
        turns.append({"role": "assistant", "content": r["response"] if r["clarifying"] or not r["memo"] else r["memo"]["headline"]})
    return results


async def run_one(name: str, sem: asyncio.Semaphore) -> None:
    messages, checks = SCENARIOS[name]
    async with sem:
        results = await run_session(messages)
    if name == "clarify_macbook":
        outcome = clarify_macbook_checks(results)
    else:
        outcome = []
        for c in checks:
            got = c(results[-1])
            outcome += got if isinstance(got, list) else [got]
    saved = {"name": name, "messages": messages, "turns": results, "checks": outcome, "ran_at": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / f"{name}.json").write_text(json.dumps(saved, indent=1, default=str), encoding="utf-8")
    print(f"{name}: {sum(c['pass'] for c in outcome)}/{len(outcome)} checks pass ({sum(r['seconds'] for r in results):.0f}s)", flush=True)


def row(saved: dict) -> str:
    last = saved["turns"][-1]
    m = last["memo"] or {}
    ref = m.get("retail_reference")
    price = f"{research._money(m['price_low'])}–{research._money(m['price_high'])}" if m else "—"
    cond = exact_conditions(last)
    ceiling = (f"{research._money(ref['price'])} ({ref.get('basis')})" if ref else (m.get("retail_status") or {}).get("status", "—")) if m else "—"
    checks = "<br>".join(f"{'PASS' if c['pass'] else '**FAIL**'} {c['check']}" for c in saved["checks"]) or "—"
    notes = []
    if not m:
        notes.append(last["response"][:140].replace("|", "/").replace("\n", " "))
    for c in saved["checks"]:
        if not c["pass"] and c["detail"]:
            notes.append(c["detail"][:140].replace("|", "/"))
    warn = (m.get("citation_check") or {}).get("warnings") or []
    notes += [f"warning: {w}" for w in warn]
    inputs = " → ".join(f"\"{x}\"" for x in saved["messages"])
    return (f"| {saved['name']} | {inputs} | {price} | {m.get('confidence', '—')} | {', '.join(cond) if cond else '—'} | "
            f"{ceiling.replace('|', '/')} | {checks} | {'<br>'.join(notes) or ''} |")


def write_summary() -> None:
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))]
    order = list(SCENARIOS)
    saved.sort(key=lambda s: order.index(s["name"]) if s["name"] in order else 99)
    n_pass = sum(all(c["pass"] for c in s["checks"]) for s in saved)
    lines = [
        "# Scenario test results",
        "",
        f"Live runs through the research pipeline (eBay Browse API, SerpApi Google Shopping with a disk cache, Gemini on Vertex). "
        f"{n_pass} of {len(saved)} scenarios pass every check. Generated by `scripts/scenario_tests.py`; full responses are in "
        f"`scenario_results/<name>.json` (not committed).",
        "",
        "| Scenario | Input | Price range | Confidence | Condition group (exact comps) | Retail ceiling | Checks | Notes |",
        "|---|---|---|---|---|---|---|---|",
        *[row(s) for s in saved],
        "",
        "## Run log",
        "",
        "| When | Scenarios | Passed every check | Live SerpApi calls |",
        "|---|---|---|---|",
    ]
    log = OUT / "runlog.jsonl"
    for line in (log.read_text(encoding="utf-8").splitlines() if log.exists() else []):
        e = json.loads(line)
        lines.append(f"| {e['when']} | {', '.join(e['names']) if len(e['names']) < len(SCENARIOS) else 'all ' + str(len(e['names']))} | "
                     f"{e['passed']} of {len(e['names'])} | {e['serpapi']} (budget {e['budget']}) |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    ap.add_argument("--serpapi-budget", type=int, default=15, help="live SerpApi calls allowed this run")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"Unknown scenarios: {unknown}. Known: {', '.join(SCENARIOS)}")
    OUT.mkdir(exist_ok=True)
    serpapi_live["budget"] = args.serpapi_budget
    guard_serpapi()
    sem = asyncio.Semaphore(args.parallel)
    await asyncio.gather(*(run_one(n, sem) for n in names))
    passed = sum(all(c["pass"] for c in json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))["checks"]) for n in names)
    with open(OUT / "runlog.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "names": names, "passed": passed,
                            "serpapi": serpapi_live["n"], "budget": serpapi_live["budget"]}) + "\n")
    write_summary()
    print(f"live SerpApi calls this run: {serpapi_live['n']} (budget {serpapi_live['budget']})")


if __name__ == "__main__":
    asyncio.run(main())
