"""The deep research pipeline behind /chat.

    Lead (intake)   is the item specific enough to price? If not, ask one clarifying question.
    Lead (research) plans, then dispatch_researchers fans out Researchers in parallel, each in a
                    fresh context. The Lead reads the findings and the gaps code found, and goes
                    back for more, up to MAX_ROUNDS rounds in all.
    Researchers     one brief each (exact_comps, condition_tiers, retail): search, drop junk titles
                    in code, then review the rest in one batch. Code turns the kept set into stats.
    Writer          turns the findings into a memo. Code renders the tables, checks every [E1]/[R1]
                    citation against the sources the tools returned, and caps the confidence.

The model never does the math: every number in a finding comes from market.py.
"""

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents import Agent, ModelSettings, RunContextWrapper, RunHooks, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from pydantic import BaseModel

import market

load_dotenv(Path(__file__).parent / ".env")

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# LiteLLM routes the call to Gemini on Vertex AI. api_key is unused: Vertex uses gcloud credentials.
MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash", api_key="unused")
# Researchers search and review titles: the cheaper, faster model is enough.
RESEARCHER_MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash-lite", api_key="unused")
# Vertex settings come from the environment (.env locally, Cloud Run env vars when deployed).
# Gemini 3.5 is served only from the global location, so that's the default: a regional
# VERTEXAI_LOCATION (e.g. us-central1) returns 404. Without VERTEXAI_PROJECT, the project comes
# from the gcloud credentials (locally) or the service account (Cloud Run).
VERTEX = {"vertex_location": os.environ.get("VERTEXAI_LOCATION") or "global"}
if os.environ.get("VERTEXAI_PROJECT"):
    VERTEX["vertex_project"] = os.environ["VERTEXAI_PROJECT"]
SETTINGS = ModelSettings(extra_args={**VERTEX, "timeout": 60, "num_retries": 2})

MAX_ROUNDS = 3  # research rounds in all: the first, plus up to 2 to fill gaps
MAX_BRIEFS_PER_ROUND = 4
MIN_COMPS = 5  # fewer kept exact comps than this is a gap, and caps confidence at low
HIGH_CONFIDENCE_COMPS = 10
# Price spread of the kept comps: IQR / median. Above TIGHT caps confidence at medium, above WIDE at low.
TIGHT_SPREAD = 0.35
WIDE_SPREAD = 0.6
RESEARCH_TIMEOUT = 130  # seconds for the whole research phase; the Writer gets what's found by then
WRITER_TIMEOUT = 75
RECOUNT_TIMEOUT = 30  # one Writer retry when its comp counts are off; 130 + 75 + 30 stays under app.py's 240
TABLE_ROWS = 12  # comps shown in the memo's table

Condition = Literal["new", "open_box", "like_new", "used", "refurbished", "for_parts"]

# Titles that mark a listing as something other than the whole item. Checked in code before the
# researcher sees the listings. Reason -> regex (case-insensitive). Tune here.
# Not plain "charging case": "AirPods Pro 2 with USB-C Charging Case" is the full item.
# Not plain "left"/"right": "left and right earbuds" is the full item.
EXCLUDE_TITLE_PATTERNS = {
    "case only": r"\bcase only\b|\bcase replacement\b|\breplacement (charging )?case\b",
    "box only": r"\bempty box\b|\bbox only\b",
    "replacement part": r"\breplacement\b",
    # one side, unless the title names both ("left and right earbuds")
    "single earbud / one side": r"\bsingle\b|^(?!.*\bleft\b.*\bright\b|.*\bright\b.*\bleft\b).*\b(left|right)\s+(side|ear|earbud|airpod|pod|bud|only)s?\b",
    "for parts / not working": r"\bfor parts\b|\bparts only\b|\bnot working\b|\bbroken\b|\bdamaged\b",
    "lot / multiple units": r"\blot\b|\bbulk\b|\bwholesale\b",
    "accessory, not the item": r"\bcompatible with\b|\bscreen protector\b|^\W*(case|cover|skin|charger|cable|strap|band)\b",
    "not genuine": r"\breplica\b|\bfake\b|\bclone\b|\bdupe\b",
    "incomplete": r"\bmissing\b|\bincomplete\b",
    "sold as-is / see description": r"\bas[- ]is\b|\bread desc",
}
# In the for_parts tier these describe the listing's condition, not a different item
PARTS_REASONS = {"for parts / not working", "replacement part"}

# Variants checked in code when the item names one (the review also checks, but misses some).
# Storage: "128GB" must appear, not followed by RAM. Carrier: "unlocked" / the carrier must appear.
STORAGE = re.compile(r"(\d+)\s*(gb|tb)\b(?!\s*(?:ram|memory|ddr))", re.I)
CARRIERS = ("unlocked", "verizon", "at&t", "t-mobile", "sprint", "cricket", "boost", "us cellular", "straight talk", "tracfone")
# A retail search whose results are at least this share resale listings, with no new offer kept,
# means no store sells the item new: "no new retail found", not retried.
RESALE_ONLY_SHARE = 0.8


# Model-tier words: "iPhone 13 Pro" is not "iPhone 13 Pro Max" or "iPhone 13". The title must have
# every tier word the product has, and none it doesn't. "plus" counts only right after the
# product's own model number ("iPhone 14 Plus"), not in "AirPods Pro 2 plus ear tips".
TIER_WORDS = re.compile(r"\b(pro|max|mini|ultra|lite)\b", re.I)


def model_exclusion(title: str, product: str) -> str | None:
    """'different model: ...' when the title's model tier differs from the product's, or None."""
    numbers = set(re.findall(r"\b\d+\b", product or ""))

    def tiers(text):
        found = {w.lower() for w in TIER_WORDS.findall(text or "")}
        if any(n in numbers for n in re.findall(r"\b(\d+)\s*plus\b", text or "", re.I)):
            found.add("plus")
        return found

    want, have = tiers(product), tiers(title)
    if want - have:
        return f"different model: not {'/'.join(sorted(w.title() for w in want - have))}"
    if have - want:
        return f"different model: {'/'.join(sorted(w.title() for w in have - want))}"
    return None


# Special editions, collabs and custom colorways price differently from the base model. Dropped
# unless the seller named the edition. Reason -> regex (case-insensitive).
EDITION_PATTERNS = {
    "special edition": r"\b(special|limited|anniversary|collector'?s?)\s+edition\b|\blimited\b",
    "SE edition": r"\bSE\b",
    "FlyEase": r"\bfly\s?ease\b",
    "custom / painted": r"\bcustom\b|\bcolorware\b|\b(hand[- ]?)?painted\b|\bskinned\b",
    "collab": r"\bcollab(oration)?\b|[a-z0-9]\s+[x\u00d7]\s+[a-z]",
}


def edition_exclusion(title: str, item_text: str) -> str | None:
    """'special edition: ...' when the title names an edition the seller's item doesn't, or None."""
    for reason, pattern in EDITION_PATTERNS.items():
        if re.search(pattern, title or "", re.I) and not re.search(pattern, item_text or "", re.I):
            return f"special edition: {reason}"
    return None


def variant_exclusion(title: str, variant: str) -> str | None:
    """Why a listing doesn't match the item's storage or carrier, or None."""
    want = {f"{n}{u}".lower() for n, u in STORAGE.findall(variant or "")}
    if want:
        have = {f"{n}{u}".lower() for n, u in STORAGE.findall(title or "")}
        if not have:
            return f"variant not stated: {', '.join(sorted(want))}"
        if not want & have:
            return f"different variant: {', '.join(sorted(have))}"
        if have - want:
            return "several variants in one listing"
    carriers = [c for c in CARRIERS if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", (variant or "").lower())]
    if carriers and not any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", (title or "").lower()) for c in carriers):
        return f"variant not stated: {carriers[0]}"
    return None


def title_exclusion(title: str, item_text: str = "", tier: str | None = None) -> str | None:
    """The first EXCLUDE_TITLE_PATTERNS reason that matches title, or None. A pattern that also
    matches the seller's own item (someone selling a 'charging case only') is skipped."""
    for reason, pattern in EXCLUDE_TITLE_PATTERNS.items():
        if tier == "for_parts" and reason in PARTS_REASONS:
            continue
        if re.search(pattern, title or "", re.I) and not re.search(pattern, item_text, re.I):
            return reason
    return None


# --- Structured outputs ---


class Item(BaseModel):
    product: str  # brand + model, e.g. "Apple AirPods Pro (2nd generation)"
    variant: str  # what changes the price within the model: storage, connector, size; "" if none
    condition: Condition
    condition_detail: str  # the seller's own words, e.g. "good, light scratches on case"
    included: str  # accessories, box, case; "" if not said
    search_query: str  # concise marketplace keywords without condition words


class Intake(BaseModel):
    ready: bool
    question: str  # the clarifying question when not ready; "" when ready
    item: Item | None
    assumptions: list[str]  # minor details assumed instead of asked


class Brief(BaseModel):
    researcher: Literal["exact_comps", "condition_tiers", "retail"]
    query: str
    conditions: list[Condition]  # eBay briefs only; [] for retail
    broadened: bool  # True when this widens an earlier search to fill a gap
    successor: bool  # retail briefs only: True when the query is the item's successor model
    instructions: str  # anything specific to look out for; "" if none


class Exclusion(BaseModel):
    id: str
    reason: str


class Memo(BaseModel):
    price_low: float
    price_high: float
    headline: str
    rationale: str
    key_factors: list[str]
    confidence: Literal["low", "medium", "high"]
    confidence_reason: str
    limitations: list[str]


# --- Run state, shared by the tools. The model never sees it. ---


class Trace(RunHooks):
    """Records every tool call, by every agent, in the order they started: /chat's tool_calls.
    type "tool" is a tool an agent called; "step" is a pipeline step run by code (intake result,
    citation check, a code fallback), listed so the trace reads end to end."""

    def __init__(self):
        self.calls: list[dict] = []
        self._open: dict[str, dict] = {}

    def record(self, agent: str, name: str, args, result, type: str = "step") -> None:
        self.calls.append({"type": type, "agent": agent, "name": name, "args": args, "result": _as_text(result)})

    async def on_tool_start(self, context, agent, tool):
        entry = {"type": "tool", "agent": agent.name, "name": tool.name, "args": _parse(getattr(context, "tool_arguments", None)), "result": None}
        self.calls.append(entry)
        self._open[getattr(context, "tool_call_id", "") or str(id(entry))] = entry

    async def on_tool_end(self, context, agent, tool, result):
        entry = self._open.pop(getattr(context, "tool_call_id", ""), None)
        if entry is None:
            self.record(agent.name, tool.name, _parse(getattr(context, "tool_arguments", None)), result, type="tool")
        else:
            entry["result"] = _as_text(result)


@dataclass
class Run:
    """One pricing run: the item, every source the tools returned, and the findings so far."""

    item: Item
    assumptions: list[str]
    trace: Trace
    sources: dict[str, dict] = field(default_factory=dict)  # "E1" / "R1" -> record
    keys: dict[str, str] = field(default_factory=dict)  # eBay itemId / retail product -> source id
    findings: list[dict] = field(default_factory=list)
    rounds: int = 0

    def register(self, prefix: str, key: str, record: dict) -> str:
        """Give a record a citation id, the same one every time the same listing comes back."""
        key = f"{prefix}:{key}"
        if key not in self.keys:
            self.keys[key] = f"{prefix}{sum(1 for k in self.keys if k.startswith(prefix + ':')) + 1}"
            self.sources[self.keys[key]] = record
        return self.keys[key]


@dataclass
class ResearcherState:
    """One researcher's working state. A fresh one per researcher, like its fresh context."""

    run: Run
    brief: Brief
    round: int
    shown: dict[str, str] = field(default_factory=dict)  # id -> condition tier, listings it reviews
    dropped: list[dict] = field(default_factory=list)  # excluded in code before the review
    reviewed: dict[str, dict] = field(default_factory=dict)  # id -> excluded in review, across every review call
    searches: list[dict] = field(default_factory=list)
    finding: dict | None = None

    @property
    def name(self) -> str:
        return f"researcher:{self.brief.researcher}"


def _parse(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _money(x) -> str:
    return "?" if x is None else f"${x:,.2f}"


def _item_text(item: Item) -> str:
    return f"{item.product} {item.variant} {item.included}"


# --- Researcher tools. Logic lives in plain functions so tests can call it directly. ---


async def do_search_ebay(state: ResearcherState, query: str, conditions: list[str]) -> dict:
    conditions = [c for c in conditions if c in market.CONDITION_IDS] or [state.run.item.condition]
    limit = 50 if len(conditions) == 1 else 25
    results = await asyncio.gather(
        *(asyncio.to_thread(market.ebay_search, query, condition=c, limit=limit) for c in conditions),
        return_exceptions=True,
    )
    listings, totals, errors = {}, {}, {}
    for tier, result in zip(conditions, results):
        if isinstance(result, Exception):
            errors[tier] = f"{type(result).__name__}: {str(result)[:200]}"
            continue
        totals[tier] = result["total"]
        lines = []
        for record in result["items"]:
            sid = state.run.register("E", record["itemId"], record)
            if sid in state.shown or any(d["id"] == sid for d in state.dropped):
                continue
            reason = title_exclusion(record["title"], _item_text(state.run.item), tier)
            if reason is None:
                reason = model_exclusion(record["title"], state.run.item.product)
            if reason is None:
                reason = edition_exclusion(record["title"], _item_text(state.run.item))
            if reason is None and not state.brief.broadened:  # a broadened brief may swap the variant
                reason = variant_exclusion(record["title"], state.run.item.variant)
            if reason is None and record["buyingOptions"] == ["AUCTION"]:
                reason = "auction: current bid, not an asking price"
            if reason:
                state.dropped.append({"id": sid, "title": record["title"], "reason": reason, "by": "code"})
                continue
            state.shown[sid] = tier
            ship = "ship ?" if record["shippingCost"] is None else f"ship {_money(record['shippingCost'])}"
            lines.append(f"{sid} | {_money(record['price'])} + {ship} | {record['condition']} | {record['title']}")
        listings[tier] = lines
    state.searches.append({"query": query, "conditions": conditions, "total_on_ebay": totals, "errors": errors})
    return {
        "query": query,
        "listings_to_review": listings,
        "dropped_in_code": [{k: d[k] for k in ("id", "title", "reason")} for d in state.dropped],
        "total_matching_on_ebay": totals,
        "errors": errors,
        "next": "Call review_listings with the listings that are not the exact item.",
    }


async def do_search_retail(state: ResearcherState, query: str) -> dict:
    try:
        result = await asyncio.to_thread(market.retail_search, query)
    except Exception as e:
        state.searches.append({"query": query, "errors": {"retail": f"{type(e).__name__}: {str(e)[:200]}"}})
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    lines = []
    for record in result["items"]:
        # The link names one offer; product_id is shared by every retailer's offer for the product
        sid = state.run.register("R", record["link"] or f"{record['product_id']}:{record['retailer']}", record)
        if record["is_resale"]:
            state.dropped.append({"id": sid, "title": record["title"], "reason": f"resale listing ({record['retailer']})", "by": "code"})
            continue
        state.shown[sid] = "new"
        sale = f" (was {record['old_price']})" if record["old_price"] else ""
        lines.append(f"{sid} | {record['price']}{sale} | {record['retailer']} | {record['title']}")
    state.searches.append({"query": query, "errors": {}})
    return {
        "query": query,
        "offers_to_review": lines,
        "dropped_in_code": [{k: d[k] for k in ("id", "title", "reason")} for d in state.dropped],
        "next": "Call review_listings with the offers that are not this product sold new.",
    }


def do_review(state: ResearcherState, exclude: list[Exclusion]) -> dict:
    """Apply the researcher's exclusions, then build the finding: every number in it from code."""
    run, brief = state.run, state.brief
    # Exclusions add up: a second search and review must not bring back what the first excluded
    for e in exclude:
        if e.id in state.shown:
            state.reviewed.setdefault(e.id, {"id": e.id, "title": run.sources[e.id]["title"], "reason": e.reason, "by": "review"})
    reviewed = list(state.reviewed.values())
    unknown = [e.id for e in exclude if e.id not in state.shown]
    excluded_ids = set(state.reviewed)
    kept = [sid for sid in state.shown if sid not in excluded_ids]
    excluded = state.dropped + reviewed

    finding = {
        "researcher": brief.researcher,
        "round": state.round,
        "queries": [s["query"] for s in state.searches],
        "broadened": brief.broadened,
        "n_found": len(state.shown) + len(state.dropped),
        "n_after_title_filter": len(state.shown),
        "n_kept": len(kept),
        "excluded_summary": dict(Counter(e["reason"] for e in excluded).most_common()),
        "excluded": excluded,
        "kept_ids": kept,
        "errors": {k: v for s in state.searches for k, v in s.get("errors", {}).items()},
    }
    if brief.researcher == "retail":
        records = [run.sources[sid] for sid in kept]
        finding["stats"] = market.price_stats([r["extracted_price"] for r in records])
        priced = [sid for sid in kept if run.sources[sid]["extracted_price"] is not None]
        finding["lowest_new_id"] = min(priced, key=lambda s: run.sources[s]["extracted_price"], default=None)
        finding["on_sale_ids"] = [sid for sid in kept if run.sources[sid]["old_price"]]
        finding["successor"] = brief.successor
        # Google Shopping gave (nearly) nothing but eBay/Poshmark/... listings: no store sells it
        # new. Retrying won't change that, so it's a result, not a gap.
        resale = sum(d["reason"].startswith("resale listing") for d in state.dropped)
        finding["only_resale_listings"] = not kept and resale > 0 and resale >= RESALE_ONLY_SHARE * finding["n_found"]
    else:
        finding["conditions"] = list(dict.fromkeys(state.shown.values())) or brief.conditions
        finding["stats"] = market.comp_stats([run.sources[sid] for sid in kept])
        if len(finding["conditions"]) > 1:
            finding["stats_by_condition"] = {
                tier: market.comp_stats([run.sources[sid] for sid in kept if state.shown[sid] == tier])
                for tier in finding["conditions"]
            }
        finding["best_offer_count"] = sum("BEST_OFFER" in run.sources[sid]["buyingOptions"] for sid in kept)
    state.finding = finding
    return {
        "n_before_review": len(state.shown),
        "n_kept": len(kept),
        "n_excluded_in_review": len(reviewed),
        "unknown_ids_ignored": unknown,
        "stats": finding.get("stats_by_condition") or finding["stats"],
    }


@function_tool
async def search_ebay(ctx: RunContextWrapper[ResearcherState], query: str, conditions: list[Condition]) -> str:
    """Search active eBay US listings, one search per condition. Listings whose titles mark them
    as accessories, parts, lots or incomplete (and auction-only listings) are dropped in code
    first; you review the rest.

    Args:
        query: eBay keywords for the product, without condition words, e.g. 'AirPods Pro 2 USB-C'.
        conditions: One or more of new, open_box, like_new, used, refurbished, for_parts.
    """
    return json.dumps(await do_search_ebay(ctx.context, query, conditions))


@function_tool
async def search_retail(ctx: RunContextWrapper[ResearcherState], query: str) -> str:
    """Search Google Shopping for current prices of the product sold new. Resale marketplace
    listings (eBay, Poshmark, ...) are dropped in code; you review the rest.

    Args:
        query: The product's name, without condition words, e.g. 'Apple AirPods Pro 2'.
    """
    return json.dumps(await do_search_retail(ctx.context, query))


@function_tool
def review_listings(ctx: RunContextWrapper[ResearcherState], exclude: list[Exclusion]) -> str:
    """Finish your review: exclude every listing that is not the exact item. Everything you don't
    exclude is kept. Returns the statistics of the kept set.

    Args:
        exclude: The listings to leave out, each with its id (e.g. 'E4') and a short reason.
    """
    return json.dumps(do_review(ctx.context, exclude))


# --- Researcher ---

researcher = Agent(
    name="researcher",
    instructions="""\
You are a pricing researcher with one brief about one secondhand item. Your job is to collect
listings and decide which ones are truly the same item as the seller's.

1. Search once with the brief's query (search_ebay with the brief's conditions, or
   search_retail for a retail brief). Search a second time only if the results are clearly
   about a different product.
2. Read every title. Call review_listings once, excluding each listing that is not the item:
   a different model or generation, a different variant that changes the price (storage,
   connector, size, carrier), an accessory or part, a bundle with extra products, several units,
   or a price that only makes sense for something else. Give each a reason of a few words,
   e.g. "different model: AirPods Pro 1", "accessory: case only", "bundle with iPhone".
   - Variant: if item.variant is not empty (storage, carrier, size) and the brief is not
     broadened, the title must state that variant. Exclude listings that don't state it
     ("variant not stated") and listings that offer several variants in one ("several
     variants: 128GB-1TB"). "Unlocked" must appear for an unlocked phone.
   - Editions: unless the item names one, exclude special or limited editions, collabs ("x Travis
     Scott"), FlyEase, SE versions and custom or painted colorways ("special edition: <which>").
   - condition_tiers: condition differences are expected. Only judge whether it's the product.
   - retail: also exclude used, refurbished or renewed offers, and offers that state a different
     variant (store titles often omit carrier: keep those). A new offer from a third-party
     seller on a retail site (e.g. "Walmart - Cellfeee") is retail: keep it. Don't exclude it because
     its price looks high or low; only when the price means it's another product (a $9 "AirPods"
     is an accessory).
   When unsure, keep the listing. Don't trust your memory of which models exist: products newer
   than your training are real.
3. Reply in one or two sentences about what you saw (common variants, data quality). Don't state
   statistics: code computes them.""",
    model=RESEARCHER_MODEL,
    model_settings=SETTINGS,
)

BRIEF_GOALS = {
    "exact_comps": "Find active eBay listings of this exact item in the seller's condition.",
    "condition_tiers": "Find active eBay listings of this product across conditions, to show how condition moves the price.",
    "retail": "Find the current price of this product sold new at retailers, and any active discounts.",
}


async def run_researcher(run: Run, brief: Brief, round_no: int) -> dict:
    """Run one researcher in a fresh context. If it fails or skips a step, code finishes the step,
    so a finding always comes back."""
    state = ResearcherState(run, brief, round_no)
    tools = [search_retail, review_listings] if brief.researcher == "retail" else [search_ebay, review_listings]
    agent = researcher.clone(name=state.name, tools=tools)
    prompt = json.dumps({
        "goal": BRIEF_GOALS[brief.researcher],
        "brief": brief.model_dump(),
        "item": run.item.model_dump(),
    }, indent=1)
    note = ""
    try:
        result = await Runner.run(agent, prompt, context=state, hooks=run.trace, max_turns=6)
        note = str(result.final_output or "")
    except Exception as e:
        note = f"Researcher stopped early ({type(e).__name__}); code finished its steps."
    if not state.searches:
        args = {"query": brief.query, "conditions": brief.conditions}
        if brief.researcher == "retail":
            run.trace.record(state.name, "search_retail (code fallback)", {"query": brief.query}, await do_search_retail(state, brief.query))
        else:
            run.trace.record(state.name, "search_ebay (code fallback)", args, await do_search_ebay(state, brief.query, brief.conditions))
    if state.finding is None:
        run.trace.record(state.name, "review_listings (code fallback: kept all)", {"exclude": []}, do_review(state, []))
    state.finding["note"] = note.strip()[:600]
    run.findings.append(state.finding)
    return state.finding


# --- Lead: research loop ---


def find_gaps(findings: list[dict]) -> list[str]:
    """What the findings can't support yet. Code decides; the Lead decides how to fill it."""
    gaps = []
    comps = [f for f in findings if f["researcher"] == "exact_comps"]
    n_comps = len({sid for f in comps for sid in f["kept_ids"]})
    if comps and n_comps < MIN_COMPS:
        gaps.append(
            f"Only {n_comps} exact comps after filtering (need {MIN_COMPS}). Broaden: an adjacent "
            "variant (drop or swap the variant words) or an adjacent condition (e.g. like_new + used)."
        )
    retail = [f for f in findings if f["researcher"] == "retail"]
    # Only resale listings came back: recorded as "no new retail found", not retried
    if retail and not any(f["n_kept"] for f in retail) and not any(f["only_resale_listings"] for f in retail):
        gaps.append(
            "No new retail price found. The item may be discontinued: try the product line without "
            "the variant, or its successor model, for a ceiling."
        )
    for f in findings:
        for where, error in f["errors"].items():
            gaps.append(f"{f['researcher']} search failed ({where}): {error}")
    return gaps


def lead_summary(finding: dict) -> dict:
    """The part of a finding the Lead needs to spot gaps; the Writer gets all of it."""
    stats = finding["stats"]
    summary = {k: finding[k] for k in ("researcher", "queries", "broadened", "n_found", "n_kept", "excluded_summary", "errors")}
    if finding["researcher"] == "retail":
        summary["median_new_price"] = stats.get("median")
        summary["lowest_new_price"] = stats.get("min")
        if finding["only_resale_listings"]:
            summary["result"] = "No new retail found: Google Shopping returned only resale listings. Don't retry retail."
    else:
        summary["conditions"] = finding["conditions"]
        summary["median_price_with_shipping"] = stats["price_with_shipping"].get("median")
    return summary


async def dispatch(run: Run, briefs: list[Brief]) -> dict:
    if run.rounds >= MAX_ROUNDS:
        return {"error": "No research rounds left. Reply DONE."}
    run.rounds += 1
    briefs = briefs[:MAX_BRIEFS_PER_ROUND]
    # The fan-out: every researcher at once, each in its own fresh context
    findings = await asyncio.gather(*(run_researcher(run, b, run.rounds) for b in briefs))
    gaps = find_gaps(run.findings)
    return {
        "round": run.rounds,
        "rounds_left": MAX_ROUNDS - run.rounds,
        "findings": [lead_summary(f) for f in findings],
        "gaps": gaps or ["None: the findings cover the item. Reply DONE."],
    }


@function_tool
async def dispatch_researchers(ctx: RunContextWrapper[Run], briefs: list[Brief]) -> str:
    """Send researchers out in parallel, one per brief, each in a fresh context. Returns a summary
    of each finding and the gaps that remain.

    Args:
        briefs: One brief per researcher, at most 4.
    """
    return json.dumps(await dispatch(ctx.context, briefs))


def default_briefs(item: Item) -> list[Brief]:
    """Round one, if the Lead never dispatches: the same plan its prompt asks for."""
    common = {"query": item.search_query, "broadened": False, "successor": False, "instructions": ""}
    return [
        Brief(researcher="exact_comps", conditions=[item.condition], **common),
        Brief(researcher="condition_tiers", conditions=["new", "open_box", "used", "for_parts"], **common),
        Brief(researcher="retail", conditions=[], **common),
    ]


lead = Agent(
    name="lead",
    instructions=f"""\
You lead a pricing research team for one secondhand item (JSON below). Researchers fetch the data;
code computes every number.

1. Plan. Call dispatch_researchers once with these three briefs in the same call:
   - exact_comps: query = concise eBay keywords for the exact model and variant (no condition
     words); conditions = [the seller's condition].
   - condition_tiers: the same query; conditions = ["new", "open_box", "used", "for_parts"].
   - retail: query = the product as a store would name it new; conditions = [].
   Add a line to a brief's instructions only if the item needs it (e.g. "must include the case").
2. Read the result's gaps. If there are none, reply DONE. Otherwise call dispatch_researchers
   again with only the briefs that fill the gaps, with broadened = true:
   - too few exact comps: an exact_comps brief with an adjacent variant or adjacent condition.
   - no retail price: the item is likely discontinued. First a retail brief with the product as a
     store would name it; if that also finds none, a retail brief for its successor model (e.g.
     AirPods Pro 3 for AirPods Pro 2) with successor = true. Don't search part numbers.
   A retail finding that says only resale listings came back is final: don't send retail again.
   successor is false on every brief except a successor-model retail brief.
   You have {MAX_ROUNDS} rounds in all. Never repeat a brief that already ran.
3. Reply DONE and nothing else: a writer writes the memo.""",
    model=MODEL,
    model_settings=SETTINGS,
    tools=[dispatch_researchers],
)

# --- Lead: intake ---

intake_agent = Agent(
    name="lead",
    instructions="""\
You are the intake step of a pricing assistant for casual resellers. Decide whether the item in
the conversation is specific enough to research a competitive asking price on eBay.

Ready needs all of:
- the product and model (brand + model/generation), e.g. "AirPods Pro 2", "iPhone 13 Pro";
- any variant that changes the price a lot: storage for phones/tablets/consoles, carrier lock
  for phones, size for clothing and shoes, screen size for laptops/TVs;
- the condition: new (sealed), open box, like new, used (any wear), refurbished, or for parts.
  "good", "works fine", "some scratches" all mean used. "never used, opened" means open_box.

If something required is missing, set ready false and ask ONE short message that asks for every
missing piece at once, e.g. "Which iPhone model and storage is it, is it unlocked or tied to a
carrier, and what condition is it in?". Don't guess required details.
Minor details the user didn't give (color, a charging-port version) don't block. Don't guess them
into the item: leave them out of variant and search_query, so the search covers every version,
and add an assumption like "Port version not given: comps include both Lightning and USB-C".
If the user has already been asked twice, go ahead with what you have, listing assumptions,
unless the product itself is still unknown.
If the message isn't about pricing an item to sell, set ready false and say in one sentence what
you do (suggest an asking price for a secondhand item from live eBay and retail data).

When ready, fill item. The latest request wins if the conversation moved to a new item.
search_query: concise marketplace keywords for the item with only the details the user gave,
e.g. "AirPods Pro 2", "iPhone 13 Pro 128GB unlocked". No condition words.""",
    model=MODEL,
    model_settings=SETTINGS,
    output_type=Intake,
)

# --- Writer ---

writer = Agent(
    name="writer",
    instructions="""\
You write a pricing memo for a casual reseller, using only the findings and sources you're given.

Price range (price_low to price_high, in dollars):
- Anchor on comps: the kept exact comps in the seller's condition. comps.stats is on
  comps.price_basis (price_with_shipping when most comps state shipping, else item_price). A competitive asking price usually sits between
  Q1 and the median: active listings skew high because overpriced items sit unsold.
- Adjust for the seller's condition detail, what's included, and the condition_tiers spread.
- Retail is a reference, not a rule. retail_reference gives resale_to_retail_ratio (median comp
  price / lowest new price) and its role:
  - "ceiling" (ratio < 1): stay below the lowest new price. If successor is true, say the price
    is the successor model's: a looser ceiling.
  - "premium" (ratio > 1): the item resells above retail (out of stock, limited, discontinued).
    Say so with the ratio, e.g. "used listings ask 1.4x the $199 retail price [R2]", and price
    from the comps alone.
  - null: no new retail found. Price from the comps and say retail couldn't be checked.
- If the comps came from a broadened search (adjacent variant or condition), say so.

Counts: comps.kept is the one comp count. Use it whenever you say how many comps there are. A
statistic over fewer listings names its subset: comps.stats covers comps.stats_n listings, so when
stats_n < kept write e.g. "median $85 across the 19 of 27 comps that list shipping". Don't quote
condition_tiers counts as comps: they cover other conditions.
Medians: the seller's condition has one median, comps.stats.median: quote that figure exactly,
everywhere. condition_tiers covers only the other conditions; when you quote one, say it's from
the broader condition search, e.g. "new listings ask a median $150 in the broader condition search".
If excluded_summary shows special editions, collabs or custom colorways, say they were left out.

Writing:
- headline: one line, the range and why, e.g. "List at $70–$78 with free shipping: just under
  the $77 median of 9 comparable used listings [E3][E8]."
- rationale: 2 to 4 sentences. key_factors: 2 to 4 bullets (condition, shipping, retail
  discounts, included accessories, best offer).
- Say how many listings were excluded and why, from comps.excluded_summary, e.g. "Excluded 4 listings
  (2 case only, 1 single earbud, 1 different model)."

Citations:
- Every price you mention ends with the tag of its source: "$93 [E3]", "new at Best Buy for
  $249.95 [R1]". For a statistic over many listings, cite 2 or 3 listings near that value:
  "median $77.49 across 9 comps [E2][E7]".
- Use only tags listed in sources. Never invent a tag or a number.

confidence: never higher than max_confidence. confidence_reason: one sentence (comp count and
price spread).
limitations: always say asking prices are not sold prices and active listings skew high; add
every data gap (few comps, no retail price, broadened search, many exclusions).
Plain, specific sentences.""",
    model=MODEL,
    model_settings=SETTINGS,
    output_type=Memo,
)


# --- Writer input, confidence, citation check, rendering ---


def comps_findings(run: Run) -> list[dict]:
    return [f for f in run.findings if f["researcher"] == "exact_comps"]


def kept_comp_ids(run: Run) -> list[str]:
    return list(dict.fromkeys(sid for f in comps_findings(run) for sid in f["kept_ids"]))


def comp_basis(run: Run) -> tuple[str, dict]:
    """The price the kept exact comps are compared on, and its stats: price_with_shipping when
    most comps state their shipping, else item_price. The Writer is told the same rule."""
    stats = market.comp_stats([run.sources[sid] for sid in kept_comp_ids(run)])
    basis = "price_with_shipping" if stats["price_with_shipping"]["n"] * 2 >= stats["item_price"]["n"] > 0 else "item_price"
    return basis, stats[basis]


def comp_summary(run: Run) -> dict:
    """The exact comps across every round, with one count: the kept comps."""
    comps = comps_findings(run)
    basis, stats = comp_basis(run)
    excluded = Counter()
    for f in comps:
        excluded.update(f["excluded_summary"])
    return {
        "kept": len(kept_comp_ids(run)),
        "found": sum(f["n_found"] for f in comps),
        "after_title_filter": sum(f["n_after_title_filter"] for f in comps),
        "excluded_summary": dict(excluded.most_common()),
        "broadened": any(f["broadened"] for f in comps),
        "queries": [q for f in comps for q in f["queries"]],
        "price_basis": basis,
        "stats_n": stats.get("n", 0),
        "stats": stats,
        "best_offer_count": sum(f.get("best_offer_count", 0) for f in comps),
        "notes": [f["note"] for f in comps if f.get("note")],
    }


COUNT_CLAIM = re.compile(r"(?<![\d$.,])(\d+)\s+(?:kept\s+|exact\s+|comparable\s+|used\s+|active\s+)*(?:comps|comparables|listings)\b", re.I)


def count_mismatches(texts: list[str], kept: int) -> list[int]:
    """Comp counts in the memo that aren't the kept count. In '19 of 27 comps' only '27 comps' is a
    claim (the total), so a subset written that way passes; exclusion counts pass too."""
    wrong = []
    for text in texts:
        for m in COUNT_CLAIM.finditer(text):
            n = int(m.group(1))
            before = text[max(0, m.start() - 25):m.start()].lower()
            if n != kept and not re.search(r"\b(excluded|excluding|dropped|removed|left out)\b", before):
                wrong.append(n)
    return wrong


def comp_spread(run: Run) -> float | None:
    """IQR / median of the kept comps: 0.2 means the middle half sits within 20% of the median."""
    st = comp_basis(run)[1]
    return round(st["iqr"] / st["median"], 3) if st.get("median") else None


def no_new_retail(run: Run) -> bool:
    return any(f["researcher"] == "retail" and f["only_resale_listings"] for f in run.findings)


def retail_reference(run: Run) -> dict | None:
    """The lowest new retail price and how resale compares to it. The item's own retail price wins
    over its successor's. role: "ceiling" if comps sit below retail, "premium" if above."""
    retail = [f for f in run.findings if f["researcher"] == "retail"]
    for successor in (False, True):
        ids = [sid for f in retail if f["successor"] == successor for sid in f["kept_ids"]]
        priced = [sid for sid in dict.fromkeys(ids) if run.sources[sid]["extracted_price"] is not None]
        if priced:
            break
    else:
        return None
    lowest = min(priced, key=lambda s: run.sources[s]["extracted_price"])
    price = run.sources[lowest]["extracted_price"]
    median = comp_basis(run)[1].get("median")
    ratio = round(median / price, 2) if median and price else None
    return {
        "id": lowest,
        "price": price,
        "retailer": run.sources[lowest]["retailer"],
        "successor": successor,
        "median_new": market.price_stats([run.sources[s]["extracted_price"] for s in priced])["median"],
        "on_sale_ids": [s for s in priced if run.sources[s]["old_price"]],
        "comp_median": median,
        "resale_to_retail_ratio": ratio,
        "role": "premium" if ratio is not None and ratio > 1 else "ceiling",
    }


def confidence_cap(run: Run) -> tuple[str, str]:
    """The highest confidence the data supports, from the kept comp count and their price spread.
    The Writer can go lower, never higher. Missing retail doesn't lower it; a successor ceiling does."""
    n_all = len(kept_comp_ids(run))
    n_exact = max((f["n_kept"] for f in comps_findings(run) if not f["broadened"]), default=0)
    spread = comp_spread(run)
    ref = retail_reference(run)
    if n_all < MIN_COMPS:
        return "low", f"only {n_all} comparable listings survived filtering"
    spread_text = f"the middle half of prices spans {spread:.0%} of the median" if spread is not None else ""
    if spread is not None and spread > WIDE_SPREAD:
        return "low", f"comp prices vary widely: {spread_text}"
    if n_exact < MIN_COMPS:
        return "medium", "the comps come partly from a broadened search (adjacent variant or condition)"
    if spread is not None and spread > TIGHT_SPREAD:
        return "medium", f"{n_all} kept comps, but {spread_text}"
    if n_exact < HIGH_CONFIDENCE_COMPS:
        return "medium", f"only {n_all} kept comps"
    if ref and ref["role"] == "ceiling" and ref["successor"]:
        return "medium", "the retail ceiling is the successor model's price"
    return "high", f"{n_all} kept comps and {spread_text}"


def source_line(sid: str, record: dict) -> str:
    if sid.startswith("E"):
        ship = "shipping ?" if record["shippingCost"] is None else f"+ {_money(record['shippingCost'])} shipping"
        return f"{sid}: {_money(record['price'])} {ship} | {record['condition']} | {record['title']}"
    sale = f" (was {record['old_price']})" if record["old_price"] else ""
    return f"{sid}: {record['price']}{sale} new | {record['retailer']} | {record['title']}"


def writer_input(run: Run, cap: tuple[str, str]) -> str:
    useful = list(dict.fromkeys(sid for f in run.findings for sid in f["kept_ids"]))
    # The exact comps go in once, merged across rounds: per-round counts are what made the Writer
    # quote 29 in one sentence and 44 in the next.
    findings = [{k: v for k, v in f.items() if k not in ("excluded", "kept_ids")} for f in run.findings if f["researcher"] != "exact_comps"]
    # The seller's own condition has one set of figures: comps. The broader condition search's
    # numbers for it (and its all-conditions stats) would give a second "used median".
    for f in findings:
        if f["researcher"] == "condition_tiers":
            f.pop("stats", None)
            f["stats_by_condition"] = {t: s for t, s in (f.get("stats_by_condition") or {}).items() if t != run.item.condition}
            f["conditions"] = [t for t in f.get("conditions", []) if t != run.item.condition]
    return json.dumps({
        "comps": comp_summary(run),
        "item": run.item.model_dump(),
        "assumptions": run.assumptions,
        "max_confidence": cap[0],
        "max_confidence_reason": cap[1],
        "comp_price_spread": comp_spread(run),
        "retail_reference": retail_reference(run),
        "no_new_retail_found": no_new_retail(run) and retail_reference(run) is None,
        "findings": findings,
        "sources": [source_line(sid, run.sources[sid]) for sid in useful],
    }, indent=1, default=str)


TAG = re.compile(r"\[\s*([ER]\d+(?:\s*[,;]\s*[ER]\d+)*)\s*\]")
BARE_ID = re.compile(r"(?<![\[\w])([ER]\d+)(?![\]\w])")


def check_citations(text: str, sources: dict) -> tuple[str, list[str], list[str]]:
    """Keep the [E1]/[R1] tags that name a real source, strip the rest.
    Returns (clean text, cited ids, unknown ids). '[E1, E2]' becomes '[E1][E2]', and a bare
    'E56' that names a real source becomes '[E56]'."""
    cited, unknown = [], []

    def fix(match):
        ids = re.split(r"\s*[,;]\s*", match.group(1))
        cited.extend(i for i in ids if i in sources)
        unknown.extend(i for i in ids if i not in sources)
        return "".join(f"[{i}]" for i in ids if i in sources)

    def bracket(match):  # a bare id that isn't a source may be a model number: leave it
        if match.group(1) in sources:
            cited.append(match.group(1))
            return f"[{match.group(1)}]"
        return match.group(1)

    text = TAG.sub(fix, text)  # first, so '[E1, E2]' is whole tags before bare ids are looked for
    return dedupe_citations(BARE_ID.sub(bracket, text)), cited, unknown


REPEAT_GAP = 40  # characters: a tag repeated this close after itself, in the same sentence, is dropped


def dedupe_citations(text: str) -> str:
    """Cite once: '[E18] ($80.00 [E18])' -> '[E18] ($80.00)', '[E1][E1]' -> '[E1]'. A repeat after
    another tag or in a later sentence stays."""
    out, pos, last_id, last_end = [], 0, None, 0
    for m in re.finditer(r"\s*\[([ER]\d+)\]", text):
        gap = text[last_end:m.start()]
        if m.group(1) == last_id and len(gap) <= REPEAT_GAP and not re.search(r"[.!?;](\s|$)", gap):
            out.append(text[pos:m.start()])
            pos = m.end()
            continue
        last_id, last_end = m.group(1), m.end()
    return "".join(out) + text[pos:]


def validate_memo(run: Run, memo: Memo, cap: tuple[str, str]) -> tuple[dict, dict]:
    """Check the Writer's memo in code: citations, confidence cap, price range sanity."""
    texts = {"headline": memo.headline, "rationale": memo.rationale, "confidence_reason": memo.confidence_reason}
    lists = {"key_factors": memo.key_factors, "limitations": memo.limitations}
    cited, unknown, clean = [], [], {}
    for k, v in texts.items():
        clean[k], c, u = check_citations(v, run.sources)
        cited += c
        unknown += u
    for k, vs in lists.items():
        clean[k] = []
        for v in vs:
            text, c, u = check_citations(v, run.sources)
            clean[k].append(text)
            cited += c
            unknown += u

    warnings = []
    excluded = {e["id"] for f in run.findings for e in f["excluded"]}
    if wrong := sorted(set(cited) & excluded - set(kept_comp_ids(run)), key=lambda s: int(s[1:])):
        warnings.append(f"cites listings that were excluded from the comps: {', '.join(wrong)}")
    low, high = sorted((round(memo.price_low, 2), round(memo.price_high, 2)))
    ref = retail_reference(run)
    if ref and ref["role"] == "ceiling" and high > ref["price"]:
        warnings.append(f"the range tops the lowest new retail price ({_money(ref['price'])} [{ref['id']}])")
    texts = [clean["headline"], clean["rationale"], clean["confidence_reason"], *clean["key_factors"]]
    if wrong := count_mismatches(texts, len(kept_comp_ids(run))):
        warnings.append(f"comp counts {sorted(set(wrong))} differ from the {len(kept_comp_ids(run))} kept comps")
    order = {"low": 0, "medium": 1, "high": 2}
    confidence = min(memo.confidence, cap[0], key=order.get)
    if confidence != memo.confidence:
        warnings.append(f"confidence lowered from {memo.confidence} to {confidence}: {cap[1]}")
        clean["confidence_reason"] = f"{clean['confidence_reason']} (Capped: {cap[1]}.)"
    clean.update(price_low=low, price_high=high, confidence=confidence)
    check = {
        "cited": sorted(set(cited), key=lambda s: (s[0], int(s[1:]))),
        "unknown_tags_stripped": unknown,
        "warnings": warnings,
    }
    return clean, check


def comp_rows(run: Run) -> list[dict]:
    """Every kept exact comp, cheapest item price first."""
    rows = []
    for sid in sorted(kept_comp_ids(run), key=lambda s: run.sources[s]["price"]):
        r = run.sources[sid]
        rows.append({"id": sid, "title": r["title"], "condition": r["condition"], "price": r["price"],
                     "shipping": r["shippingCost"], "total": r["totalPrice"], "url": r["itemWebUrl"], "image": r.get("image"),
                     "best_offer": "BEST_OFFER" in r["buyingOptions"]})
    return rows


def spread(rows: list, k: int = TABLE_ROWS) -> list:
    """k rows evenly spaced through rows (sorted), so a table shows the whole range."""
    if len(rows) <= k:
        return rows
    return [rows[round(i * (len(rows) - 1) / (k - 1))] for i in range(k)]


def tier_rows(run: Run) -> list[dict]:
    """Median price per condition, best condition first. The seller's own condition uses the exact
    comps (the memo's canonical figures, source "exact_comps"); the other conditions come from the
    broader condition search (source "condition_search"), where a tier searched again in a later
    round keeps its first result."""
    rows = {}
    if kept_comp_ids(run):
        st = comp_basis(run)[1]
        rows[run.item.condition] = {"condition": run.item.condition, "n": len(kept_comp_ids(run)), "median": st["median"],
                                    "q1": st["q1"], "q3": st["q3"], "source": "exact_comps"}
    for f in run.findings:
        if f["researcher"] != "condition_tiers":
            continue
        for tier, s in (f.get("stats_by_condition") or {}).items():
            st = s["price_with_shipping"] if s["price_with_shipping"]["n"] else s["item_price"]
            if st["n"] and tier not in rows:
                rows[tier] = {"condition": tier, "n": st["n"], "median": st["median"], "q1": st["q1"], "q3": st["q3"], "source": "condition_search"}
    order = list(market.CONDITION_IDS)
    return sorted(rows.values(), key=lambda r: order.index(r["condition"]) if r["condition"] in order else len(order))


def retail_rows(run: Run) -> list[dict]:
    ids = list(dict.fromkeys(sid for f in run.findings if f["researcher"] == "retail" for sid in f["kept_ids"]))
    rows = [{"id": s, "retailer": run.sources[s]["retailer"], "title": run.sources[s]["title"], "price": run.sources[s]["extracted_price"],
             "old_price": run.sources[s]["extracted_old_price"], "url": run.sources[s]["link"]} for s in ids]
    return sorted((r for r in rows if r["price"] is not None), key=lambda r: r["price"])[:8]


def render(run: Run, memo: dict, rows: dict, cited: list[str]) -> tuple[str, dict]:
    """The memo as markdown, plus the sources it cites. Every table number comes from code."""
    comps = comps_findings(run)
    n_found = sum(f["n_found"] for f in comps)
    n_title = sum(f["n_after_title_filter"] for f in comps)
    n_kept = len(kept_comp_ids(run))
    table = spread(rows["comps"])
    excluded = Counter()
    for f in comps:
        excluded.update(f["excluded_summary"])
    lines = [
        f"## Suggested asking price: {_money(memo['price_low'])} – {_money(memo['price_high'])}",
        "*A competitive asking price from active listings, not a prediction of what it will sell for.*",
        "",
        f"**{memo['headline']}**",
        "",
        f"**Confidence: {memo['confidence']}.** {memo['confidence_reason']}",
        "",
        memo["rationale"],
        "",
        "### Key factors",
        *[f"- {k}" for k in memo["key_factors"]],
        "",
        f"### Comparable listings ({n_kept} kept)",
        f"{n_found} found, {n_title} after the title filter, {n_kept} after review."
        + (f" Showing {len(table)} spread across the range, cheapest first." if len(table) < n_kept else ""),
        "",
        "| Source | Listing | Condition | Price | Shipping | Total |",
        "|---|---|---|---|---|---|",
        *[f"| [{r['id']}] | {r['title'][:70]} | {r['condition']} | {_money(r['price'])} | {_money(r['shipping'])} | {_money(r['total'])} |" for r in table],
    ]
    if excluded:
        lines += ["", "Excluded from comps: " + ", ".join(f"{n} {reason}" for reason, n in excluded.most_common()) + "."]
    if rows["tiers"]:
        lines += ["", "### Price by condition (active listings)", "| Condition | Listings | Median | Middle half |", "|---|---|---|---|"]
        lines += [f"| {t['condition']} | {t['n']} | {_money(t['median'])} | {_money(t['q1'])} – {_money(t['q3'])} |" for t in rows["tiers"]]
    lines += ["", "### Retail (new)"]
    ref = retail_reference(run)
    if ref:
        model = "the successor model's " if ref["successor"] else ""
        x = ref["resale_to_retail_ratio"]
        if ref["role"] == "premium":
            lines += [f"Resale premium: comps ask {x:.2f}x {model}lowest new price ({_money(ref['comp_median'])} vs "
                      f"{_money(ref['price'])} [{ref['id']}]), so the price comes from comps alone.", ""]
        elif x is not None:
            lines += [f"Resale-to-retail ratio {x:.2f}: comps sit below {model}lowest new price of {_money(ref['price'])} "
                      f"[{ref['id']}], which acts as a ceiling.", ""]
    if rows["retail"]:
        lines += ["| Source | Retailer | Price | Was |", "|---|---|---|---|"]
        lines += [f"| [{r['id']}] | {r['retailer']} | {_money(r['price'])} | {_money(r['old_price']) if r['old_price'] else ''} |" for r in rows["retail"]]
    elif no_new_retail(run):
        lines += ["No new retail found: Google Shopping returned only resale listings, so the item is likely discontinued."]
    else:
        lines += ["No current new retail price found: the item may be discontinued."]
    lines += ["", "### Limitations", *[f"- {x}" for x in memo["limitations"]]]

    shown = list(dict.fromkeys(cited + [r["id"] for r in table] + [r["id"] for r in rows["retail"]]))
    shown.sort(key=lambda s: (s[0], int(s[1:])))
    sources = {}
    for sid in shown:
        r = run.sources[sid]
        url = r.get("itemWebUrl") or r.get("link")
        sources[sid] = {"title": r["title"], "url": url, "price": r.get("price") if sid[0] == "E" else r.get("extracted_price"),
                        "detail": r.get("condition") if sid[0] == "E" else r.get("retailer")}
    lines += ["", "### Sources", *[f"- [{sid}] [{s['title'][:80]}]({s['url']}): {s['detail']}, {_money(s['price'])}" for sid, s in sources.items()]]
    return "\n".join(lines), sources


async def write_memo(run: Run) -> tuple[str, dict | None]:
    if not kept_comp_ids(run):
        searched = "; ".join(f"{f['researcher']}: {', '.join(f['queries'])}" for f in run.findings) or "nothing"
        retail = retail_reference(run)
        extra =f" For reference, it sells new for {_money(retail['price'])} at {retail['retailer']} [{retail['id']}]." if retail else ""
        return (
            f"I couldn't find enough active eBay listings of this item to suggest an asking price (searched {searched}).{extra} "
            "Try a more common name for the item, or check the model name.",
            None,
        )
    cap = confidence_cap(run)
    prompt = writer_input(run, cap)
    draft = (await asyncio.wait_for(Runner.run(writer, prompt, hooks=run.trace, max_turns=2), WRITER_TIMEOUT)).final_output
    kept = len(kept_comp_ids(run))
    if wrong := count_mismatches([draft.headline, draft.rationale, draft.confidence_reason, *draft.key_factors], kept):
        # One retry with the exact problem named; if it fails or times out, the first draft stands
        note = (f"\n\nRevise your memo: it gives comp counts {sorted(set(wrong))}, but there are {kept} kept comps. "
                f"Say {kept} comps, and write any subset as 'N of {kept}' (e.g. '21 of {kept} list shipping'). Change nothing else.")
        try:
            revised = (await asyncio.wait_for(Runner.run(writer, prompt + note, hooks=run.trace, max_turns=2), RECOUNT_TIMEOUT)).final_output
            still = count_mismatches([revised.headline, revised.rationale, revised.confidence_reason, *revised.key_factors], kept)
            run.trace.record("writer", "recount", {"counts_found": sorted(set(wrong)), "kept": kept}, {"revised": True, "counts_still_off": sorted(set(still))})
            if len(still) < len(wrong):
                draft = revised
        except Exception as e:
            run.trace.record("writer", "recount", {"counts_found": sorted(set(wrong)), "kept": kept}, {"revised": False, "error": type(e).__name__})
    memo, check = validate_memo(run, draft, cap)
    rows = {"comps": comp_rows(run), "tiers": tier_rows(run), "retail": retail_rows(run)}
    markdown, sources = render(run, memo, rows, check["cited"])
    run.trace.record("writer", "validate_citations", {"tags_in_memo": check["cited"] + check["unknown_tags_stripped"]}, check)
    excluded = Counter()
    for f in comps_findings(run):
        excluded.update(f["excluded_summary"])
    return markdown, {
        "item": run.item.model_dump(),
        "assumptions": run.assumptions,
        **memo,
        "comps": rows["comps"],
        "comps_found": sum(f["n_found"] for f in comps_findings(run)),
        "comps_after_title_filter": sum(f["n_after_title_filter"] for f in comps_findings(run)),
        "comps_kept": len(kept_comp_ids(run)),
        "excluded_summary": dict(excluded.most_common()),
        "condition_tiers": rows["tiers"],
        "retail": rows["retail"],
        "retail_reference": retail_reference(run),
        "no_new_retail_found": no_new_retail(run) and retail_reference(run) is None,
        "comp_price_spread": comp_spread(run),
        "sources": sources,
        "citation_check": check,
        "rounds": run.rounds,
    }


# --- The whole pipeline ---


def transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)


async def answer(turns: list[dict], clarifications: int) -> dict:
    """turns: the session's {role, content} messages, the latest user message last.
    Returns {response, tool_calls, memo, clarifying}."""
    trace = Trace()
    prompt = f"{transcript(turns)}\n\n(Clarifying questions already asked: {clarifications})"
    intake = (await Runner.run(intake_agent, prompt, hooks=trace, max_turns=2)).final_output
    trace.record("lead", "assess_item", {"conversation": transcript(turns)}, intake.model_dump())
    if not intake.ready or intake.item is None:
        question = intake.question or "What item is it (brand and model), and what condition is it in?"
        return {"response": question, "tool_calls": trace.calls, "memo": None, "clarifying": True}

    run = Run(item=intake.item, assumptions=intake.assumptions, trace=trace)
    try:
        await asyncio.wait_for(Runner.run(lead, run.item.model_dump_json(indent=1), context=run, hooks=trace, max_turns=8), RESEARCH_TIMEOUT)
    except Exception as e:  # a timeout or a Lead that won't stop: write from what's found
        trace.record("lead", "research_stopped", {}, {"reason": f"{type(e).__name__}: {str(e)[:200]}"})
    if not run.findings:  # the Lead never dispatched: run its plan in code
        briefs = default_briefs(run.item)
        trace.record("lead", "dispatch_researchers (code fallback)", {"briefs": [b.model_dump() for b in briefs]}, await dispatch(run, briefs))
    response, memo = await write_memo(run)
    return {"response": response, "tool_calls": trace.calls, "memo": memo, "clarifying": False}


# --- Draft my listing (POST /draft) ---

EBAY_TITLE_MAX = 80
DRAFT_TIMEOUT = 30
CONDITION_LABELS = {"new": "New", "open_box": "Open Box", "like_new": "Like New", "used": "Used",
                    "refurbished": "Refurbished", "for_parts": "For Parts"}


class Listing(BaseModel):
    title: str
    description: str


lister = Agent(
    name="writer",
    instructions=f"""\
You draft an eBay listing for a casual reseller from the facts given. Use only those facts.

title: at most {EBAY_TITLE_MAX} characters. Brand, model, variant (storage, carrier, size), the key
included accessory, then condition, e.g. "Apple AirPods Pro 2nd Gen with USB-C Charging Case - Used
Good". No ALL CAPS, emojis, "L@@K" or words like "rare" or "mint" unless the seller said them.
No [brackets] in the title: leave out details you don't know.

description: plain text, under 120 words. A one-sentence intro, then short "- " bullet lines for
Condition (the seller's own words), Included, and Shipping. Never invent specs, accessories,
battery health, defects or history. Where a buyer would expect a detail the facts don't give, write
a bracketed placeholder for the seller to fill in, e.g. "[add battery health]", "[describe any
scratches; add photos]". Don't mention the price.""",
    model=MODEL,
    model_settings=SETTINGS,
    output_type=Listing,
)


def listing_facts(memo: dict) -> dict:
    item = memo.get("item") or {}
    comps = memo.get("comps") or []
    free = sum(c.get("shipping") == 0 for c in comps)
    return {
        "product": item.get("product", ""),
        "variant": item.get("variant", ""),
        "condition": CONDITION_LABELS.get(item.get("condition"), item.get("condition", "")),
        "condition_detail": item.get("condition_detail", ""),
        "included": item.get("included", ""),
        "assumptions": memo.get("assumptions") or [],
        "most_comps_ship_free": bool(comps) and free * 2 >= len(comps),
    }


def tidy_description(text: str) -> str:
    """Real line breaks, one '- Label:' bullet per line. The model sometimes writes newlines as a
    literal backslash-n, or runs the bullets together."""
    text = (text or "").replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+-\s+(?=[A-Z][\w ]{1,20}:)", "\n- ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def clip_title(title: str) -> str:
    """eBay's 80-character limit, cut at a word boundary. Placeholders don't belong in a title."""
    title = re.sub(r"\[[^\]]*\]", "", title or "")
    title = " ".join(title.split())
    if len(title) <= EBAY_TITLE_MAX:
        return title
    return title[:EBAY_TITLE_MAX + 1].rsplit(" ", 1)[0].rstrip(" -,")


def template_listing(facts: dict) -> Listing:
    """No model needed: the same facts in a fixed shape."""
    parts = [facts["product"], facts["variant"]]
    if facts["included"]:
        parts.append(f"with {facts['included']}")
    parts.append(f"- {facts['condition']}")
    lines = [
        f"{facts['product']} {facts['variant']}".strip() + f", {facts['condition'].lower()}.",
        f"- Condition: {facts['condition_detail'] or '[describe the condition; add photos]'}",
        f"- Included: {facts['included'] or '[list what is included]'}",
        "- Shipping: " + ("Free shipping." if facts["most_comps_ship_free"] else "[shipping cost and service]"),
    ]
    return Listing(title=" ".join(p for p in parts if p), description="\n".join(lines))


async def draft_listing(memo: dict) -> dict:
    """An eBay-style title and description at the top of the suggested range (room for offers)."""
    facts = listing_facts(memo)
    price = memo.get("price_high")
    source = "model"
    try:
        result = await asyncio.wait_for(Runner.run(lister, json.dumps(facts, indent=1), max_turns=1), DRAFT_TIMEOUT)
        listing = result.final_output
    except Exception:
        listing, source = template_listing(facts), "template"
    return {
        "title": clip_title(listing.title),
        "description": tidy_description(listing.description),
        "price": price,
        "price_note": "The top of the suggested range, leaving room for Best Offer.",
        "source": source,
    }
