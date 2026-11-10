# Rummage

Rummage suggests a competitive asking price for a secondhand item. It uses live eBay listings and current retail prices. The result is an asking price, not a prediction of what the item will sell for. It is built for casual resellers on eBay, Depop, Mercari and Facebook Marketplace. IEOR 4570 Project 2, Deep Research path, solo (jz3980).

Live: https://resale-agent-git-410194130748.us-central1.run.app/ (needs Columbia login)

## Data

| Source | What it gives |
|---|---|
| eBay Browse API, `item_summary/search` | Live listings: price, shipping, condition and sub-grade, seller, dates, URLs |
| SerpApi Google Shopping | Current retail prices. Only big-box and brand stores count as retail. Marketplace and third-party sellers are excluded in code. |

Nothing is stored in the repo. Both are live APIs called at query time.

To run locally, copy `.env.example` to `.env` and fill in `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` (eBay production keyset), `SERPAPI_KEY` and `VERTEXAI_PROJECT` (leave `VERTEXAI_LOCATION=global`). Gemini runs on Vertex AI, so log in once with `gcloud auth application-default login`. Then:

```
uv sync
uv run python app.py
```

Open http://localhost:8000.

## How it works

- The Lead checks that the item is specific enough: the model, any variant that changes the price, and the condition. If not, it asks one clarifying question.
- The Lead sends three Researchers in parallel, each with a fresh context: `exact_comps`, `condition_tiers` and `retail`.
- Each Researcher filters in code first (accessories, parts, lots, wrong variants). Then the LLM reviews what's left and gives a reason for each exclusion. All stats are computed in code.
- Each eBay seller is counted once, so one seller with many listings can't set the median.
- Code checks for gaps (fewer than 5 comps, no retail price, a failed search). The Lead goes back for only what's missing, up to 3 rounds.
- The Writer produces a short memo where every price is cited [E#] or [R#]. Code checks every citation and caps the confidence.
- It says so when the data can't answer: retail not found (discontinued, boutique-only, or no mainstream price), too few comps, or authenticity that can't be verified. Off-topic questions get a one-line scope reply.

Clarification example. Input: "i want sell my shoes". Reply:

> What brand and model are the shoes, what size are they, and what condition are they in?

Gap loop example, from the scenario test run at 2026-11-09 19:51 ("Nintendo Switch OLED, used, good condition"). After round 1, code reported this gap:

> Retail search for 'Nintendo Switch OLED console' returned only resale listings. Try once more with exactly 'Nintendo Switch OLED'; not a successor model.

In round 2 the Lead sent only a retail brief for "Nintendo Switch OLED". It found $399.99 at Nintendo's store, and code reported no more gaps.

## Design choices

- Rules in code, judgment by the LLM. Filters (storage, carrier, size, bundles, editions), stats, citations and confidence caps are code. The LLM reviews ambiguous listings and writes the memo.
- Each eBay seller counts once, at its median listing. One PS5 seller's six identical listings were setting the median before this.
- Condition wording maps to eBay's groups ("new without tags" → New without box, "doesn't turn on" → For parts), and sub-grades (Pre-owned Good/Excellent) weight the comps.
- Retail is a reference, not always a ceiling. If used comps ask more than retail (collectibles), the memo says so instead of capping the price.

## Sample queries

Run on Nov 9, 2026. Live prices change, so a new run will give somewhat different numbers. "Expected" is the range the live comps support (between the lower end of the middle half and the median), with the confidence code allows.

**"Used AirPods Pro 2, good condition, with charging case"**

| | |
|---|---|
| Expected answer | About $71-80, medium confidence (prices spread widely: middle half $71-105) |
| Where it comes from | eBay `item_summary/search`, used: 50 found, 25 kept from 24 sellers, median $79.99 with shipping. Google Shopping: Best Buy $249.95 new. |
| Agent's answer | $70-80, medium. "List at $70-80, around the $80 median asking price of similar used listings." "New is cheap: it's $250 at Best Buy [R21]..." "12 of 25 used comps accept Best Offer, like this $80 pair with a Lightning case [E25]." |
| Baseline answer | USB-C version $130-160, Lightning version $100-130. New retail $249. No sources, data "as of 2024". |

Live listings change daily; reruns on Nov 9 gave $70-85 with medium to high confidence depending on price spread.

**"iPhone 13 Pro 128GB unlocked, used with light scratches"**

| | |
|---|---|
| Expected answer | About $300-320, high confidence (tight spread: middle half $300-335) |
| Where it comes from | eBay `item_summary/search`, used: 50 found, 27 kept from 26 sellers, median $319.99 with shipping. Google Shopping: Walmart $413.99 new. |
| Agent's answer | $300-320, high. "List at $300-320 with free shipping, around the $320 median asking price of similar used listings." "27 used listings from 26 sellers mostly ask $300 [E11] to $340 [E10]." |
| Baseline answer | eBay or Swappa $400-460, private sale $380-440, trade-in $250-300. No sources, data "as of 2024/2025". |

**"Chanel Classic Flap medium, black caviar leather, gold hardware, used in good condition"**

| | |
|---|---|
| Expected answer | About $6,500-7,100, high confidence, anchored on the Pre-owned Good median ($7,036) because the seller said good condition. All-used comps run lower (median $6,506). No online retail price, since Chanel sells new only in boutiques. |
| Where it comes from | eBay `item_summary/search`, used: 50 found, 22 kept from 17 sellers, median $6,505.68 with shipping (Pre-owned Good median $7,036). Google Shopping: no retail, boutique-only brand. |
| Agent's answer | $6,500-7,100, high. "Chanel sells new only in boutiques; list at $6,500-$7,100 with free shipping, near the $7,036 median for used Good condition bags." "14 of 22 comps accept Best Offer, including a black caviar model listed at $6,499 [E2]." Watch out: comps aren't verified authentic. |
| Baseline answer | Good condition $6,500-8,500. Says retail is $10,800. No sources. |

The baseline is the Lead's model (`vertex_ai/gemini-3.5-flash`) with no tools and no system prompt: `uv run python scripts/baseline.py`. Its answers are trimmed to the price sentence; [full answers here](docs/baseline_answers.md).

Without tools, the model overprices the electronics by $30-140 (AirPods $100-160 vs live $70-80; iPhone $400-460 vs live $300-320) and cites nothing. For Chanel its range overlaps but is wider, quotes an unsourced $10,800 retail price, and has no listings behind it.

## Limitations

- These are asking prices, not sold prices, and active listings skew high.
- Results vary from run to run (live listings and LLM wording).
- Retail prices are whatever Google Shopping returns and are not independently verified.
- Session memory is in-process, so Cloud Run runs as one instance with session affinity.
