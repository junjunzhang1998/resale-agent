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

# --- What kind of item it is: picks the title filters, variant checks and retail wording ---

CATEGORY_PATTERNS = {  # first match wins
    "earbuds": r"\bair\s?pods\b|\bearbuds?\b|\bgalaxy buds\b|\bpixel buds\b|\bbeats\b|\bheadphones?\b|\bearphones?\b",
    "phone": r"\biphone\b|\bgalaxy (s|z|note|a)\s?\d|\bpixel \d|\bsmartphone\b|\bphone\b",
    "laptop": r"\bmacbook\b|\blaptop\b|\bnotebook\b|\bthinkpad\b|\bchromebook\b|\bsurface laptop\b|\bxps\b|\bzenbook\b",
    "console": r"\bps[45]\b|\bplaystation\b|\bxbox\b|\bnintendo switch\b|\bswitch (oled|lite)\b|\bsteam deck\b|\bdualsense\b",
    "watch": r"\bwatch\b|\brolex\b|\bsubmariner\b|\bdatejust\b|\bdaytona\b|\bomega\b|\bpatek\b|\baudemars\b|\bcartier\b",
    "apparel": r"\bleggings?\b|\bjeans\b|\bpants\b|\bdress\b|\b(t-)?shirt\b|\bjacket\b|\bcoat\b|\bhoodie\b|\bsweater\b|\bskirt\b"
               r"|\bshorts\b|\bblouse\b|\bsweatshirt\b|\bjoggers\b|\blululemon\b|\bsneakers?\b|\bshoes?\b|\bboots?\b"
               r"|\bair force 1\b|\bair jordan\b|\bjordan \d|\bdunk\b|\byeezy\b|\bair max\b|\bnew balance \d",
    "handbag": r"\b(hand)?bags?\b|\bpurse\b|\btote\b|\bclutch\b|\bflap\b|\bsatchel\b|\bcrossbody\b|\bhobo\b|\bwoc\b"
               r"|\bwallet on chain\b|\bbirkin\b|\bkelly\b|\bspeedy\b|\bneverfull\b|\bpochette\b|\bbackpack\b",
}
LUXURY_BRANDS = r"\bchanel\b|\bherm[eè]s\b|\blouis vuitton\b|\bgucci\b|\bprada\b|\bdior\b|\bc[eé]line\b|\bfendi\b|\bsaint laurent\b|\bysl\b" \
                r"|\bbottega veneta\b|\bgoyard\b|\bloewe\b|\bbalenciaga\b|\bvalentino\b|\brolex\b|\bcartier\b|\bomega\b|\bpatek\b|\baudemars\b"
# First-party retail is boutique-only: no online new price exists, which is not "discontinued"
BOUTIQUE_ONLY_BRANDS = {"Chanel": r"\bchanel\b", "Hermès": r"\bherm[eè]s\b", "Rolex": r"\brolex\b", "Goyard": r"\bgoyard\b",
                        "Patek Philippe": r"\bpatek\b", "Audemars Piguet": r"\baudemars\b"}


def item_profile(item: "Item") -> dict:
    """{category, luxury, boutique_brand} from the item's own words. Code decides, not the model."""
    text = f"{item.product} {item.variant} {item.search_query}"
    category = next((c for c, p in CATEGORY_PATTERNS.items() if re.search(p, text, re.I)), "general")
    luxury = bool(re.search(LUXURY_BRANDS, text, re.I))
    if luxury and category == "general":
        category = "handbag"  # "Chanel Classic Flap" without the word bag
    boutique = next((b for b, p in BOUTIQUE_ONLY_BRANDS.items() if re.search(p, text, re.I)), None)
    # a one-of-a-kind item (a 1970s handmade quilt) has no exact comps, only similar items
    similar = category == "general" and bool(re.search(ONE_OF_A_KIND, f"{text} {item.condition_detail}", re.I))
    return {"category": category, "luxury": luxury, "boutique_brand": boutique, "similar_items": similar}


ONE_OF_A_KIND = r"\bvintage\b|\bantique\b|\bhand[- ]?made\b|\bhand[- ]?(stitched|quilted|crafted|knit(ted)?|sewn|carved|painted|thrown)\b" \
                r"|\bone[- ]of[- ]a[- ]kind\b|\bfolk art\b|\bcustom[- ]made\b|\bhomemade\b|\bheirloom\b|\b(18|19)\d0s\b"
SIMILAR_NOTE = ("These comps are similar items, not exact matches: a one-of-a-kind piece has no identical listings, "
                "so its price depends on its own pattern, maker and condition.")


# Titles that mark a listing as something other than the whole item. Checked in code before the
# researcher sees the listings. (label, regex (case-insensitive), categories it applies to; None =
# all). A pattern for another category never fires: "Chanel Single Flap" is not a single earbud.
# Not plain "charging case": "AirPods Pro 2 with USB-C Charging Case" is the full item.
# Not plain "left"/"right": "left and right earbuds" is the full item.
EXCLUDE_TITLE_PATTERNS = [
    ("case only", r"\bcase only\b|\bcase replacement\b|\breplacement (charging )?case\b", {"earbuds"}),
    ("box only", r"\bempty box\b|\bbox only\b", None),
    ("dust bag / box only", r"\bdust ?bag only\b|\bonly (the )?dust ?bag\b|^\W*(\w+\s+){0,2}dust ?bag\b", {"handbag"}),
    ("replacement part", r"\breplacement\b", None),
    # one side, unless the title names both ("left and right earbuds")
    ("single earbud / one side", r"\bsingle\b|^(?!.*\bleft\b.*\bright\b|.*\bright\b.*\bleft\b).*\b(left|right)\s+(side|ear|earbud|airpod|pod|bud|only)s?\b", {"earbuds"}),
    ("for parts / not working", r"\bfor parts\b|\bparts only\b|\bnot working\b|\bbroken\b|\bdamaged\b", None),
    ("lot / multiple units", r"\blot\b|\bbulk\b|\bwholesale\b", None),
    ("accessory, not the item", r"\bcompatible with\b|\bscreen protector\b|^\W*(case|cover|skin|charger|cable|strap|band)\b", {"earbuds", "phone", "watch", "laptop", "console", "apparel", "general"}),
    ("accessory, not the bag", r"\borgani[sz]er\b|\b(purse |bag )?insert\b|\bshaper\b|\bstrap only\b|\bchain only\b|\bcharm\b"
                               r"|\bfor (chanel|herm[eè]s|louis vuitton|lv|gucci|dior)\b|^\W*(strap|chain|insert)\b", {"handbag"}),
    ("not genuine", r"\breplica\b|\bfake\b|\bclone\b|\bdupe\b|\binspired\b|\blook[- ]?alikes?\b|\bnot (authentic|genuine|real)\b|\bunbranded\b", None),
    # luxury titles say "Style A01112" (a style number): only "style" on its own marks a lookalike
    ("not genuine", r"\bstyle\b(?!\s*(?:#|no\b|number|code|[a-z]{0,2}\d))", {"handbag", "watch"}),
    ("incomplete", r"\bmissing\b|\bincomplete\b", None),
    ("sold as-is / see description", r"\bas[- ]is\b|\bread desc", None),
]
# In the for_parts tier these describe the listing's condition, not a different item
PARTS_REASONS = {"for parts / not working", "replacement part"}

# Handbags: size, leather and hardware change the price. A title that states a different one is
# another variant; one that doesn't say is left to the review (bag titles often omit hardware).
BAG_VARIANTS = {
    "size": {"mini": r"\bmini\b", "small": r"\bsmall\b", "medium": r"\bmedium\b|\bm/l\b", "jumbo": r"\bjumbo\b|\blarge\b", "maxi": r"\bmaxi\b"},
    "leather": {"caviar": r"\bcaviar\b", "lambskin": r"\blamb(skin)?\b", "patent": r"\bpatent\b", "calfskin": r"\bcalf(skin)?\b",
                "tweed": r"\btweed\b", "denim": r"\bdenim\b", "suede": r"\bsuede\b", "velvet": r"\bvelvet\b"},
    "hardware": {"gold": r"\bghw\b|\b(light |pale )?gold[- ]?(tone[- ]?)?(hardware|hw)\b|\b24k\b",
                 "silver": r"\bshw\b|\bsilver[- ]?(tone[- ]?)?(hardware|hw)\b|\bpalladium\b",
                 "ruthenium": r"\bruthenium\b|\brhw\b|\bso black\b"},
}


def bag_variant_exclusion(title: str, variant_text: str) -> str | None:
    """'different size: jumbo' when the title states another size, leather or hardware, or None."""
    norm = lambda s: re.sub(r"\bmedium\s*[/-]\s*large\b", "medium", s or "", flags=re.I)  # noqa: E731  the M/L flap is medium
    title, variant_text = norm(title), norm(variant_text)
    for dim, options in BAG_VARIANTS.items():
        want = {o for o, p in options.items() if re.search(p, variant_text, re.I)}
        have = {o for o, p in options.items() if re.search(p, title, re.I)}
        if want and have and not want & have:
            return f"different {dim}: {', '.join(sorted(have))}"
    return None

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


RAM_SIZES = {"4gb", "8gb", "12gb", "16gb", "18gb", "24gb", "32gb", "36gb", "48gb", "64gb"}
RAM = re.compile(r"(\d+)\s*gb\s*(?:of\s*)?(?:ram|memory|unified memory|ddr\d?|lpddr\d?x?)\b", re.I)
CHIP = re.compile(r"\bM([1-5])\b(?!\s*(?:pro|max|ultra))(?:\s*chip)?|\b(?:core\s*)?(i[3579])\b", re.I)


def laptop_variant_exclusion(title: str, variant: str) -> str | None:
    """Laptops: chip, RAM and storage. Titles often write RAM and storage as bare sizes ("M2 8GB
    256GB"): with the item's RAM known, a size up to 64GB that isn't storage is the RAM."""
    title, variant = title or "", variant or ""
    want_chip = {c.lower() for pair in CHIP.findall(variant) for c in pair if c}
    have_chip = {c.lower() for pair in CHIP.findall(title) for c in pair if c}
    if want_chip and have_chip and not want_chip & have_chip:
        return f"different chip: {', '.join(sorted('M' + c if c.isdigit() else c for c in have_chip))}"
    want_ram = {f"{n}gb" for n in RAM.findall(variant)}
    want_sto = {f"{n}{u}".lower() for n, u in STORAGE.findall(variant)}
    sizes = {f"{n}{u}".lower() for n, u in re.findall(r"(\d+)\s*(gb|tb)\b", title, re.I)}
    have_ram = {f"{n}gb" for n in RAM.findall(title)} | {s for s in sizes - want_sto if s in RAM_SIZES and want_ram}
    have_sto = sizes - have_ram - {s for s in sizes if s in RAM_SIZES and s in want_ram}
    if want_ram and have_ram and not want_ram & have_ram:
        return f"different RAM: {', '.join(sorted(have_ram))}"
    if want_sto:
        if not have_sto:
            return f"variant not stated: {', '.join(sorted(want_sto))}"
        if not want_sto & have_sto:
            return f"different storage: {', '.join(sorted(have_sto))}"
        if have_sto - want_sto:
            return "several variants in one listing"
    return None


# Consoles: a bundle with extra games or controllers prices as more than the console. The seller's
# own extras ("with 1 controller") are fine; disc and digital editions are different models.
CONSOLE_BUNDLE = r"\bbundle\b|\b(2|3|4|two|three|four)\s*(x\s*)?controllers?\b|\b(with|w/|\+|plus|&)\s*(\d+\s+)?(games?|titles)\b|\bgames? included\b|\b\d+\s+games\b"


def _seller_words(item: "Item") -> str:
    return item.condition_detail or ""


MODEL_CODE = re.compile(r"\b([A-Za-z]{1,3})(\d{1,4})\b")
# Appliance product lines within a model: a Dyson V8 Animal isn't a V8 Absolute
PRODUCT_LINES = r"animal|absolute|motorhead|origin|advanced|extra|total clean|fluffy|detect|outsize|complete|cyclone"


def model_code_exclusion(title: str, product: str) -> str | None:
    """'Dyson V8' vs a 'Dyson V10' title, 'Galaxy S23' vs 'S24', 'PS5' vs 'PS4': same letters,
    another number. A title that also names the product's own code is left to the review. And when
    the product names a line ('V8 Animal'), a title naming another line is another model."""
    title, product = title or "", product or ""
    have = {(p.lower(), n) for p, n in MODEL_CODE.findall(title)}
    for prefix, number in MODEL_CODE.findall(product):
        same = {n for p, n in have if p == prefix.lower()}
        if same and number not in same:
            return f"different model: {prefix.upper()}{'/'.join(sorted(same, key=int))}"
    mine = {m.lower() for m in re.findall(rf"\b({PRODUCT_LINES})\b", product, re.I)}
    theirs = {m.lower() for m in re.findall(rf"\b({PRODUCT_LINES})\b", title, re.I)}
    if mine and theirs and not mine & theirs:
        return f"different model line: {', '.join(sorted(theirs))}"
    return None


# Apparel and shoes: a stated size or color that isn't the seller's is another item
APPAREL_SIZE = re.compile(r"\b(?:size|sz)\.?\s*:?\s*(\d{1,2}(?:\.5)?|xxs|xs|s|m|l|xl|xxl|2xl|3xl)\b"
                          r"|\b(?:women'?s|womens|wmns|men'?s|mens)\s+(\d{1,2}(?:\.5)?)\b|\b(\d{1,2}(?:\.5)?)\s*(?=(?:w|women'?s|womens|wmns)\b)", re.I)  # lookahead: "'07 Womens 8.5" reads both
COLORS = ("black", "white", "navy", "blue", "grey", "gray", "green", "red", "pink", "purple", "brown", "beige", "tan", "cream",
          "ivory", "olive", "yellow", "orange", "burgundy", "maroon", "teal", "khaki", "silver", "gold", "lavender", "coral")
COLOR_ALIASES = {"gray": "grey", "blk": "black", "onyx": "black"}


def _sizes(text: str) -> set[str]:
    return {next(g for g in m if g).lower() for m in APPAREL_SIZE.findall(text or "")}


def _colors(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", (text or "").lower())
    return {COLOR_ALIASES.get(w, w) for w in words if COLOR_ALIASES.get(w, w) in COLORS}


def apparel_variant_exclusion(title: str, variant: str, product: str = "") -> str | None:
    """Like storage for phones: a size the title states that isn't the seller's ('size 4' for a
    size 6), or colors that leave out the seller's ('Navy' for black; 'Black/White' is fine).
    A title that doesn't say is left to the review. The product's own numbers aren't sizes:
    'Air Force 1 Womens' is not a size 1."""
    product_numbers = set(re.findall(r"\d+", product or ""))
    want, have = _sizes(variant), _sizes(title) - product_numbers
    if want and have and not want & have:
        return f"different size: {', '.join(sorted(have))}"
    want_c, have_c = _colors(variant), _colors(title)
    if want_c and have_c and not want_c & have_c:
        return f"different color: {', '.join(sorted(have_c))}"
    return None


def console_exclusion(title: str, item_text: str) -> str | None:
    title, item_text = title or "", item_text or ""
    if re.search(CONSOLE_BUNDLE, title, re.I) and not re.search(CONSOLE_BUNDLE, item_text, re.I):
        return "bundle with extras"
    # console revisions: a PS5 Slim isn't the original PS5, a Switch OLED isn't the original Switch
    for word in ("slim", "oled"):
        mine, theirs = re.search(rf"\b{word}\b", item_text, re.I), re.search(rf"\b{word}\b", title, re.I)
        if mine and not theirs:
            return f"different model: not {word.upper() if word == 'oled' else word.title()}"
        if theirs and not mine:
            return f"different model: {word.upper() if word == 'oled' else word.title()}"
    for mine, other in (("disc", "digital"), ("digital", "disc")):
        if re.search(rf"\b{mine}\b", item_text, re.I) and re.search(rf"\b{other}\b", title, re.I) and not re.search(rf"\b{mine}\b", title, re.I):
            return f"different edition: {other}"
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


def title_exclusion(title: str, item_text: str = "", tier: str | None = None, category: str = "earbuds") -> str | None:
    """The first EXCLUDE_TITLE_PATTERNS reason for the item's category that matches title, or None.
    A pattern that also matches the seller's own item (someone selling a 'charging case only') is
    skipped. category defaults to earbuds, the patterns' original scope."""
    for reason, pattern, categories in EXCLUDE_TITLE_PATTERNS:
        if categories is not None and category not in categories:
            continue
        if tier == "for_parts" and reason in PARTS_REASONS:
            continue
        if re.search(pattern, title or "", re.I) and not re.search(pattern, item_text, re.I):
            return reason
    return None


# --- Condition sub-grades: eBay's "Pre-owned - Good" etc. ---

GRADES = ["Poor", "Fair", "Good", "Very Good", "Excellent"]  # worst to best


def listing_grade(condition: str | None) -> str | None:
    """'Pre-owned - Excellent' -> 'Excellent'. Plain 'Used' / 'Pre-owned' has no grade: None."""
    text = (condition or "").lower()
    if "like new" in text:
        return "Excellent"
    if "acceptable" in text:
        return "Fair"
    for grade in ("Very Good", "Excellent", "Good", "Fair", "Poor"):  # "very good" before "good"
        if re.search(rf"\b{grade.lower()}\b", text):
            return grade
    return None


# The seller's words -> a grade, checked in order: an explicit grade word first, then wear words
SELLER_GRADE_PATTERNS = [
    ("Very Good", r"\bvery good\b|\bgreat (condition|shape)\b"),
    ("Excellent", r"\bexcellent\b|\blike new\b|\bmint\b|\bpristine\b|\bflawless\b|\bbarely used\b|\bno (signs of )?wear\b"),
    ("Poor", r"\bpoor\b|\bheavily (worn|damaged|used)\b|\bcracked\b"),
    ("Fair", r"\bfair\b|\bheavy (wear|scratch(es)?|scuff(s|ing)?)\b|\bwell[- ]worn\b|\bnoticeable (wear|scratch(es)?|scuffs?)\b|\bdeep scratch|\bdents?\b"),
    ("Good", r"\bgood\b|\blight(ly)? (used|worn|wear|scratch(es)?|scuff(s|ing)?)\b|\bminor (wear|scratch(es)?|scuffs?)\b|\bnormal wear\b"
             r"|\bworks (fine|great|perfectly)\b|\bsome (scratch(es)?|wear|scuffs?)\b"),
]


def seller_grade(item: "Item") -> str | None:
    """The eBay sub-grade the seller's description matches, or None (not a used item, or vague)."""
    if item.condition == "like_new":
        return "Excellent"
    if item.condition != "used":
        return None
    for grade, pattern in SELLER_GRADE_PATTERNS:
        if re.search(pattern, item.condition_detail or "", re.I):
            return grade
    return None


# The seller's words -> eBay's condition group, checked in code after intake (the model tends to
# call "new without tags" like_new, which searches Pre-owned - Excellent). Unworn but not sealed
# or tagged is eBay's "New without box / tags" (1500): open_box. Worn at all is pre-owned.
OPEN_BOX_WORDS = re.compile(r"\bnew without (the )?(tags?|box)\b|\bnew,? (no|w/o) (tags?|box)\b|\bnwot\b|\bnwob\b|\bnew other\b"
                            r"|\bnever (been )?(worn|used)\b|\bunworn\b|\bunused\b|\bopened,? (but )?(never|not) used\b", re.I)
LIKE_NEW_WORDS = re.compile(r"\bworn (once|twice|a (few|couple of) times)\b|\bused (once|twice)\b|\blike[- ]new\b|\bexcellent\b"
                            r"|\bbarely (worn|used)\b|\bmint\b", re.I)
NEW_WORDS = re.compile(r"\bnew with (tags?|box)\b|\bnwt\b|\bnwb\b|\bsealed\b|\bbrand new in box\b|\bdeadstock\b", re.I)


NOT_WORKING_WORDS = re.compile(r"\b(does ?n[o']?t|wo ?n[o']?t|will not|can ?n[o']?t|does not|won't|doesn't)\s+(turn on|power on|boot|charge|work|start)\b"
                               r"|\bnot (working|turning on|powering on)\b|\bfor parts\b|\bparts only\b|\bdead\b|\bno power\b|\bbricked\b", re.I)
WORKS_WORDS = re.compile(r"\b(but |still |fully )?(works|working|functional|functions)( fine| great| perfectly| normally)?\b", re.I)


def drop_guessed_variant(item: "Item", user_text: str) -> "Item":
    """Storage or carrier the seller never said ("1TB" assumed for a PS5) is a guess: the strict
    variant check would then drop every listing that doesn't state it. Leave it out instead."""
    said = re.sub(r"\s+", "", (user_text or "").lower())
    guessed = [f"{n}{u}" for n, u in STORAGE.findall(item.variant or "") if f"{n}{u}".lower() not in said]
    guessed += [c for c in CARRIERS if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", (item.variant or "").lower()) and c not in (user_text or "").lower()]
    if not guessed:
        return item

    def strip(text: str) -> str:
        for g in guessed:
            if g[-2:].lower() in ("gb", "tb"):  # "1TB" also as "1 TB"
                pattern = rf"\b{re.escape(g[:-2])}\s*{g[-2:]}\b"
            else:
                pattern = rf"(?<!\w){re.escape(g)}(?!\w)"
            text = re.sub(pattern, " ", text or "", flags=re.I)
        return " ".join(text.replace(" ,", ",").split()).strip(" ,")

    return item.model_copy(update={"variant": strip(item.variant), "search_query": strip(item.search_query)})


def map_condition(item: "Item", text: str = "") -> str:
    """The condition group for the seller's own words (item.condition_detail plus their messages).
    Wear wins over "new" words ("like new, worn once" is pre-owned); then unworn-but-not-sealed
    phrases are open_box; then new-with-tags is new. Otherwise the intake's choice stands."""
    words = f"{item.condition_detail} {text}"
    # Doesn't work -> for parts. Damaged but working ("cracked back but works") is used, not for parts.
    if NOT_WORKING_WORDS.search(words):
        return "for_parts"
    if item.condition == "for_parts" and WORKS_WORDS.search(words):
        return "used"
    if item.condition in ("refurbished", "for_parts"):
        return item.condition
    if LIKE_NEW_WORDS.search(words) and not re.search(r"\bnever (been )?(worn|used)\b|\bunworn\b", words, re.I):
        return "like_new"
    if OPEN_BOX_WORDS.search(words):
        return "open_box"
    if NEW_WORDS.search(words) and item.condition in ("open_box", "like_new"):
        return "new"
    return item.condition


def grade_weight(listing: str | None, seller: str) -> float:
    """How much a comp counts toward the seller's grade: the same grade 1, one step off 0.5, two or
    more 0.2, ungraded ('Used') 0.5."""
    if listing is None:
        return 0.5
    return {0: 1.0, 1: 0.5}.get(abs(GRADES.index(listing) - GRADES.index(seller)), 0.2)


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
    item = state.run.item
    category = item_profile(item)["category"]
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
            reason = title_exclusion(record["title"], _item_text(item), tier, category)
            # Bag sizes (mini, jumbo) are checked as a variant below, not as model tiers
            if reason is None and category not in ("handbag", "apparel"):
                reason = model_exclusion(record["title"], item.product) or model_code_exclusion(record["title"], item.product)
            # a for-parts item that doesn't work prices against listings that don't work either
            if reason is None and tier == "for_parts" and item.condition == "for_parts" and NOT_WORKING_WORDS.search(_seller_words(item)) \
                    and WORKS_WORDS.search(record["title"] or "") and not NOT_WORKING_WORDS.search(record["title"] or ""):
                reason = "works (yours doesn't): not comparable for parts"
            if reason is None:
                reason = edition_exclusion(record["title"], _item_text(item))
            if reason is None and category == "console":
                reason = console_exclusion(record["title"], f"{_item_text(item)} {item.condition_detail}")
            if reason is None and not state.brief.broadened:  # a broadened brief may swap the variant
                # laptop titles write RAM as a bare size ("8GB 256GB"): the storage check would call it two variants
                reason = (laptop_variant_exclusion(record["title"], f"{item.product} {item.variant}") if category == "laptop"
                          else apparel_variant_exclusion(record["title"], item.variant, item.product) if category == "apparel"
                          else variant_exclusion(record["title"], item.variant))
                if reason is None and category == "handbag":
                    reason = bag_variant_exclusion(record["title"], f"{item.product} {item.variant}")
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


THIRD_PARTY = "third-party seller, not a mainstream retailer"
RETAIL_SIZE = re.compile(r"\s*\b(?:size|sz)\.?\s*\d+(?:\.5)?\s*(?:[wm]\b)?", re.I)


def retail_query(query: str) -> str:
    """Stores sell every size at one price, and Google Shopping answers 'size 7' with resale
    listings (an AF1 search came back all eBay and Poshmark): drop explicit sizes."""
    return " ".join(RETAIL_SIZE.sub(" ", query or "").split()) or query


def retail_title_exclusion(title: str, item: "Item", successor: bool = False) -> str | None:
    """The code checks eBay listings get, for a store offer: bundles, editions and stated
    variants. Store titles often leave a variant out, so only a different stated one counts. A
    successor-model search skips the model and chip checks (it's a different model by design)."""
    category = item_profile(item)["category"]
    if re.search(r"\b(refurbished|renewed|pre-?owned|used|open box)\b", title or "", re.I):
        return "not new: refurbished / used offer"
    if category == "console" and (reason := console_exclusion(title, f"{_item_text(item)} {item.condition_detail}")):
        return reason
    if not successor and category not in ("handbag",) and (reason := model_exclusion(title, item.product)):
        return reason
    if category == "laptop":
        reason = laptop_variant_exclusion(title, f"{'' if successor else item.product} {item.variant}")
        return reason if reason and not reason.startswith("variant not stated") else None
    if category == "handbag":
        return bag_variant_exclusion(title, f"{item.product} {item.variant}")
    reason = variant_exclusion(title, item.variant)
    return reason if reason and reason.startswith("different") else None


async def do_search_retail(state: ResearcherState, query: str) -> dict:
    query = retail_query(query)
    try:
        result = await asyncio.to_thread(market.retail_search, query)
    except Exception as e:
        state.searches.append({"query": query, "errors": {"retail": f"{type(e).__name__}: {str(e)[:200]}"}})
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    lines = []
    boutique = item_profile(state.run.item)["boutique_brand"]
    for record in result["items"]:
        # The link names one offer; product_id is shared by every retailer's offer for the product
        sid = state.run.register("R", record["link"] or f"{record['product_id']}:{record['retailer']}", record)
        if record["is_resale"]:
            state.dropped.append({"id": sid, "title": record["title"], "reason": f"resale listing ({record['retailer']})", "by": "code"})
            continue
        # Chanel, Hermès, Rolex sell new only in their boutiques: any other "new" offer is a reseller
        if boutique and not re.search(BOUTIQUE_ONLY_BRANDS[boutique], record["retailer"] or "", re.I):
            state.dropped.append({"id": sid, "title": record["title"], "reason": f"resale listing ({record['retailer']}; {boutique} sells new only in boutiques)", "by": "code"})
            continue
        if reason := retail_title_exclusion(record["title"], state.run.item, state.brief.successor):
            state.dropped.append({"id": sid, "title": record["title"], "reason": reason, "by": "code"})
            continue
        # A marketplace or independent seller's new price isn't a retail ceiling: decided here, not by the reviewer
        if not is_major_retailer(record["retailer"], state.run.item.product):
            state.dropped.append({"id": sid, "title": record["title"], "reason": f"{THIRD_PARTY} ({record['retailer']})", "by": "code"})
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
            state.reviewed.setdefault(e.id, {"id": e.id, "title": run.sources[e.id]["title"], "reason": review_reason(e.reason), "by": "review"})
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
        # new offers exist, but only from third-party sellers: no mainstream price, and a retry won't find one
        finding["no_mainstream"] = not kept and not finding["only_resale_listings"] and any(d["reason"].startswith(THIRD_PARTY) for d in state.dropped)
    else:
        finding["conditions"] = list(dict.fromkeys(state.shown.values())) or brief.conditions
        finding["stats"] = market.comp_stats([run.sources[sid] for sid in kept])
        if len(finding["conditions"]) > 1:
            finding["stats_by_condition"] = {
                tier: market.comp_stats([run.sources[sid] for sid in kept if state.shown[sid] == tier])
                for tier in finding["conditions"]
            }
            finding["tier_ids"] = {tier: [sid for sid in kept if state.shown[sid] == tier] for tier in finding["conditions"]}
        finding["best_offer_count"] = sum("BEST_OFFER" in run.sources[sid]["buyingOptions"] for sid in kept)
    state.finding = finding
    # n_found = n_dropped_in_code + n_before_review, and n_before_review = n_kept + n_excluded_in_review
    return {
        "n_found": finding["n_found"],
        "n_dropped_in_code": len(state.dropped),
        "n_before_review": len(state.shown),
        "n_kept": len(kept),
        "n_excluded_in_review": len(reviewed),
        "unknown_ids_ignored": unknown,
        "stats": finding.get("stats_by_condition") or finding["stats"],
    }


def review_reason(reason: str) -> str:
    """The reviewer's own short reason, tidied: one line, lower-case start, at most 60 characters.
    Kept as written rather than forced into a code bucket that may not fit the item."""
    reason = " ".join((reason or "").split()).rstrip(".") or "not the exact item"
    reason = reason[0].lower() + reason[1:] if not reason[:2].isupper() else reason
    return reason if len(reason) <= 60 else reason[:57].rsplit(" ", 1)[0] + "…"


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
     variant (store titles often omit carrier: keep those). Code has already dropped
     third-party marketplace sellers. Don't exclude a store offer because
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
    if brief.researcher == "exact_comps" and not brief.broadened:
        brief = brief.model_copy(update={"conditions": [run.item.condition]})  # the mapped group, whatever the Lead sent
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


def find_gaps(findings: list[dict], item: "Item | None" = None) -> list[str]:
    """What the findings can't support yet. Code decides; the Lead decides how to fill it."""
    gaps = []
    boutique = item_profile(item)["boutique_brand"] if item else None
    comps = [f for f in findings if f["researcher"] == "exact_comps"]
    n_comps = len({sid for f in comps for sid in f["kept_ids"]})
    if comps and n_comps < MIN_COMPS:
        gaps.append(
            f"Only {n_comps} exact comps after filtering (need {MIN_COMPS}). Broaden: an adjacent "
            "variant (drop or swap the variant words) or an adjacent condition (e.g. like_new + used)."
        )
    retail = [f for f in findings if f["researcher"] == "retail"]
    # Only resale listings came back: recorded as "no new retail found", not retried
    # A boutique-only brand (Chanel, Hermès, Rolex) has no online retail price to find: not a gap
    # ... nor are searches that errored: that's reported below as a failure, not as "discontinued"
    if retail and not boutique and not any(f["n_kept"] for f in retail) and not any(f["only_resale_listings"] for f in retail) \
            and not all(f["errors"] for f in retail) and not any(f.get("no_mainstream") for f in retail):
        gaps.append(
            "No new retail price found. The item may be discontinued: try the product line without "
            "the variant, or its successor model, for a ceiling."
        )
    # Only resale listings for one short query ("Dyson V8") may just be the query: one more try with
    # the store's full product name ("Dyson V8 Cordless Vacuum"). After two, it's final.
    retail_queries = [q for f in retail for q in f["queries"]]
    if retail and not boutique and not any(f["n_kept"] for f in retail) and any(f["only_resale_listings"] for f in retail) \
            and len(set(retail_queries)) < 2:
        gaps.append(f"Retail search for '{retail_queries[0] if retail_queries else ''}' returned only resale listings. Try once more with "
                    "the full product name a store would use (brand, model and product type, e.g. 'Dyson V8 Cordless Vacuum'); "
                    "not a successor model.")
    for f in findings:
        for where, error in f["errors"].items():
            if PERMANENT_ERROR.search(error):  # quota, credentials, a test budget: a retry fails the same way
                gaps.append(f"{f['researcher']} search failed ({where}): {api_reason(error)} Retrying won't help: don't send {f['researcher']} again.")
            else:
                gaps.append(f"{f['researcher']} search failed ({where}): {error}")
    return gaps


PERMANENT_ERROR = re.compile(r"quota|rate-limited|rejected the credentials|credentials missing|key missing|budget", re.I)


def api_reason(error: str) -> str:
    """'ApiError: SerpApi quota used up ...' -> 'SerpApi quota used up ...' (the message says what to do)."""
    return re.sub(r"^\w*Error:\s*", "", error or "").strip()


def lead_summary(finding: dict) -> dict:
    """The part of a finding the Lead needs to spot gaps; the Writer gets all of it."""
    stats = finding["stats"]
    summary = {k: finding[k] for k in ("researcher", "queries", "broadened", "n_found", "n_kept", "excluded_summary", "errors")}
    if finding["researcher"] == "retail":
        summary["median_new_price"] = stats.get("median")
        summary["lowest_new_price"] = stats.get("min")
        if finding["only_resale_listings"]:
            summary["result"] = "No new retail found: Google Shopping returned only resale listings. See the gaps for whether to try once more."
        elif finding.get("no_mainstream"):
            summary["result"] = "No mainstream retail price: only third-party sellers list it new. Don't retry retail or search a successor."
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
    gaps = find_gaps(run.findings, run.item)
    out = {
        "round": run.rounds,
        "rounds_left": MAX_ROUNDS - run.rounds,
        "findings": [lead_summary(f) for f in findings],
        "gaps": gaps or ["None: the findings cover the item. Reply DONE."],
    }
    if boutique := item_profile(run.item)["boutique_brand"]:
        out["retail_note"] = f"Retail is final: {boutique} sells new only in its boutiques, so there is no online retail price. Don't send retail again."
    return out


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
   - retail: query = the product as a store would name it new, without a clothing or shoe size
     (stores price every size the same); conditions = [].
   Add a line to a brief's instructions only if the item needs it (e.g. "must include the case").
2. Read the result's gaps. If there are none, reply DONE. Otherwise call dispatch_researchers
   again with only the briefs that fill the gaps, with broadened = true:
   - too few exact comps: an exact_comps brief with an adjacent variant or adjacent condition.
   - no retail price: the item is likely discontinued. First a retail brief with the product as a
     store would name it; if that also finds none, a retail brief for its successor model (e.g.
     AirPods Pro 3 for AirPods Pro 2) with successor = true. Don't search part numbers.
   A retail finding that says only resale listings came back is final once two queries have
   tried (the gaps say when to try once more): then don't send retail again.
   So is a retail_note in the result (a boutique-only brand such as Chanel): no successor search.
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
  for phones, size for clothing, shoes and handbags (e.g. mini, small, medium, jumbo for a Chanel
  flap), chip/processor, RAM and storage for laptops (e.g. "M2, 8GB RAM, 256GB"), screen size
  for TVs. A laptop's screen size is minor when the chip, RAM and storage are given: don't ask,
  add an assumption ("Screen size not given: comps include 13- and 15-inch") and leave it out;
- the condition: new (sealed), open box, like new, used (any wear), refurbished, or for parts.
  "good", "works fine", "some scratches" all mean used. "never used, opened", "new without tags",
  "new without box", "never worn" and "unworn" mean open_box. "worn once", "like new" and
  "excellent" mean like_new.

If something required is missing, set ready false and ask ONE short message that asks for every
missing piece at once, e.g. "Which iPhone model and storage is it, is it unlocked or tied to a
carrier, and what condition is it in?". Don't guess required details.
Minor details the user didn't give (color, a charging-port version) don't block. Don't guess them
into the item: leave them out of variant and search_query, so the search covers every version,
and add an assumption like "Port version not given: comps include both Lightning and USB-C".
Handbags: size, leather (caviar, lambskin) and hardware (gold, silver) all change the price. Put
every one the user gave in variant and search_query, e.g. variant "medium, caviar, gold hardware".
Size is required; leather and hardware not given become assumptions.
Appliances with product lines (a Dyson V8 Animal, Absolute or Origin): put the line in product
when given; when it isn't, don't ask: add an assumption ("V8 line not given: comps mix Animal,
Absolute and other V8 models").
Each assumption is one short sentence about one detail. Authenticity is never an assumption: don't
add one about it (the memo covers it).
If the user has already been asked twice, go ahead with what you have, listing assumptions,
unless the product itself is still unknown.
If the message names more than one item to sell, set ready false and ask which one to price
first (one at a time), with that item's missing details in the same question.
If the message isn't about pricing an item to sell, set ready false and say in one sentence what
you do (suggest an asking price for a secondhand item from live eBay and retail data).
Firearms, ammunition, gun parts and other weapons: set ready false and politely decline (you
don't price them; eBay doesn't allow them). Don't ask for details about them.

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
- comps.grade, when present, is the seller's eBay sub-grade (Pre-owned - Good, ...). Every comp
  stays in; comps.anchor_median weights them toward that grade. Anchor on comps.anchor_median.
  When comps.grade.note is set, the grade median differs from the used median: code adds that
  note to the key factors, so don't restate it.
- Adjust for the seller's condition detail, what's included, and the condition_tiers spread.
- Handbags (item_profile.category "handbag"): size, leather and hardware change the price; name
  the ones the comps share, e.g. "medium caviar flaps with gold hardware".
- Retail is a reference, not a rule. retail_reference.price is the retail ceiling: the median of
  major retailers, each store counted once at its lowest price (retail_reference.basis says which), robust to one cheap offer.
  resale_to_retail_ratio is median comp price / that ceiling. Quote the ceiling with its basis,
  e.g. "$110 (median of 4 major retailers)", with a tag only when retail_reference.id is set.
  retail_reference.lowest_note, when set, is one cheaper deal: you may mention it as a deal, but
  it is never the ceiling. The role:
  - "ceiling" (ratio < 1): stay below the ceiling. If successor is true, say the price
    is the successor model's: a looser ceiling.
  - "premium" (ratio > 1): the item resells above retail (out of stock, limited, discontinued).
    Say so with the ratio, e.g. "used listings ask 1.4x the $199 retail price", and price
    from the comps alone. For an everyday item (not vintage, collectible or luxury) code adds a
    limitation flagging it and caps confidence at medium: lean toward Q1, not the median.
  - null: no retail price. retail_status.note says why, in code's words: use its meaning and
    never contradict it. "boutique_only" means the brand sells new only in its boutiques: say
    so, and never call the item discontinued. "discontinued" / "not_found": say retail couldn't
    be checked online. "search_failed": the search errored; say the retail price couldn't be
    checked, and don't call the item discontinued or unavailable. "no_mainstream": only third-party
    sellers list it new; say there's no mainstream retail price and don't quote theirs.
- new_vs_retail, when present, means sealed-new listings on eBay sit far below retail. Code adds
  its sentence to the key factors: don't restate it, but you may rely on it in the rationale.
- If the comps came from a broadened search (adjacent variant or condition), say so.
- item_profile.similar_items true (vintage, handmade, antique): the comps are similar items, not
  the same piece. Code opens the rationale with a sentence saying so: don't repeat it. Describe
  what the comps share (e.g. "vintage handmade queen quilts") and keep the range wide.

Counts: comps.kept is the one comp count. Use it whenever you say how many comps there are. A
statistic over fewer listings names its subset: comps.stats covers comps.stats_n listings, so when
stats_n < kept write e.g. "median $85 across the 19 of 27 comps that list shipping". Don't quote
condition_tiers counts as comps: they cover other conditions.
Medians: the seller's condition has one median, comps.stats.median, and comps.median_phrase says
how to name it: quote that figure exactly, with its basis, everywhere. condition_tiers covers only
the other conditions; when you quote one, say it's from the broader condition search, e.g. "new
listings ask a median $150 in the broader condition search".
Exclusions: end the rationale with comps.exclusion_sentence, copied exactly. Write no other
exclusion counts anywhere. You may add, without numbers, that special editions or custom
colorways were left out.

Writing:
- headline: one line, the range and why, e.g. "List at $70–$78 with free shipping: just under
  the $77 median of 9 comparable used listings [E3][E8]."
- rationale: 2 to 4 sentences, then comps.exclusion_sentence. key_factors: 2 to 4 bullets
  (condition, shipping, retail discounts, included accessories, best offer).

Citations:
- A listing's own price ends with its tag: "$93 [E3]", "new at Best Buy for $249.95 [R1]". Tag a
  listing only when the number is that listing's price, total or shipping.
- A statistic (median, Q1, weighted median, a range endpoint taken from one) gets no listing tag:
  name its basis instead, e.g. "median $77.49 across 9 comps", "Q1 of $75.25 across the 22 of 32
  comps that list shipping".
- Use only tags listed in sources. Never invent a tag or a number.
- Best Offer: mention it only when comps.best_offer_share is 0.3 or more.

confidence: never higher than max_confidence. confidence_reason: one sentence (comp count and
price spread).
limitations: always say asking prices are not sold prices and active listings skew high; add
every data gap (few comps, no retail price, broadened search, many exclusions). Leave
authenticity out: for luxury items code adds that limitation.
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


def exclusion_groups(summary: dict) -> list[dict]:
    """'different model: Max' and 'different model: Pro 1' -> one group, the details kept. The chips
    on the page and the memo's exclusion sentence both come from this, so they can't disagree."""
    groups: dict[str, dict] = {}
    for reason, n in summary.items():
        label, _, detail = reason.partition(":")
        g = groups.setdefault(label.strip(), {"label": label.strip(), "n": 0, "details": []})
        g["n"] += n
        if detail.strip():
            g["details"].append(f"{n}× {detail.strip()}")
    return sorted(groups.values(), key=lambda g: (-g["n"], g["label"]))


def comp_funnel(run: Run) -> dict:
    """The exact comps' funnel, each listing counted once across rounds (a later round re-finds the
    same listings). found = dropped in code + excluded in review + kept, exactly."""
    kept = kept_comp_ids(run)
    out: dict[str, dict] = {}
    for f in comps_findings(run):
        for e in f["excluded"]:
            if e["id"] not in kept:
                out.setdefault(e["id"], e)  # its first reason
    excluded = list(out.values())
    summary = Counter(e["reason"] for e in excluded)
    by_code = sum(e["by"] == "code" for e in excluded)
    return {
        "found": len(kept) + len(excluded),
        "after_title_filter": len(kept) + len(excluded) - by_code,
        "kept": len(kept),
        "excluded": len(excluded),
        "excluded_summary": dict(summary.most_common()),
        "excluded_groups": exclusion_groups(dict(summary.most_common())),
    }


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def exclusion_sentence(funnel: dict) -> str:
    """The memo's one sentence about exclusions, from the same groups as the chips."""
    if not funnel["excluded"]:
        return f"No listings were excluded: all {funnel['found']} matched the item."
    parts = [f"{g['n']} {g['label']}" for g in funnel["excluded_groups"]]
    return f"Excluded {funnel['excluded']} of {funnel['found']} listings: {_join(parts)}."


def comp_median(run: Run) -> dict:
    """The one median of the seller's condition, and how to name it: over all kept comps, or over
    the N of M that list shipping. The memo, the condition table and the chart all use this."""
    basis, stats = comp_basis(run)
    n, of = stats.get("n", 0), len(kept_comp_ids(run))
    if basis == "price_with_shipping" and n < of:
        phrase = f"median {_money(stats.get('median'))} (price + shipping) across the {n} of {of} comps that list shipping"
    elif basis == "price_with_shipping":
        phrase = f"median {_money(stats.get('median'))} (price + shipping) across all {of} comps"
    else:
        phrase = f"median {_money(stats.get('median'))} (item price) across all {of} comps"
    return {"basis": basis, "n": n, "of": of, "median": stats.get("median"), "q1": stats.get("q1"), "q3": stats.get("q3"), "phrase": phrase}


def _basis_value(record: dict, basis: str) -> float | None:
    return record.get("totalPrice") if basis == "price_with_shipping" else record.get("price")


BEST_OFFER_SHARE = 0.3  # below this share of kept comps accepting offers, Best Offer isn't a factor


def best_offer_count(run: Run) -> int:
    return sum("BEST_OFFER" in run.sources[s].get("buyingOptions", []) for s in kept_comp_ids(run))


def best_offer_share(run: Run) -> float:
    n = len(kept_comp_ids(run))
    return round(best_offer_count(run) / n, 2) if n else 0.0


def grade_summary(run: Run) -> dict | None:
    """The seller's eBay sub-grade (Pre-owned - Good, ...) and the comps weighted toward it. Every
    comp stays in the set; only the weights change. None when there is no grade to match."""
    grade, ids = seller_grade(run.item), kept_comp_ids(run)
    if not grade or not ids:
        return None
    basis, stats = comp_basis(run)
    rows = [(s, _basis_value(run.sources[s], basis), listing_grade(run.sources[s]["condition"])) for s in ids]
    rows = [r for r in rows if r[1] is not None]
    if not any(g for _, _, g in rows):
        return None  # no comp states a grade
    matching = [s for s, _, g in rows if g == grade]
    m_median = market.price_stats([v for s, v, _ in rows if s in matching]).get("median")
    used = stats.get("median")
    weighted = market.weighted_median([v for _, v, _ in rows], [grade_weight(g, grade) for _, _, g in rows])
    differs = len(matching) >= 3 and bool(used) and abs(m_median - used) / used >= 0.03
    cond = CONDITION_LABELS.get(run.item.condition, run.item.condition)
    note = None
    if differs:
        note = (f"{cond} median {_money(used)} across {comp_median(run)['phrase'].split(' across ', 1)[1]}; the {len(matching)} listed "
                f"Pre-owned - {grade} (your grade) ask a median {_money(m_median)}, "
                f"so the range leans toward that grade.")
    return {
        "grade": grade,
        "label": f"Pre-owned - {grade}",
        "n_matching": len(matching),
        "n": len(rows),
        "by_grade": dict(Counter(g or "ungraded" for _, _, g in rows).most_common()),
        "used_median": used,
        "matching_median": m_median,
        "weighted_median": weighted,
        "differs": differs,
        "note": note,
    }


def anchor_median(run: Run) -> float | None:
    """The price the range is anchored on: the grade-weighted median when the seller's grade is
    known, else the plain median."""
    g = grade_summary(run)
    return g["weighted_median"] if g else comp_basis(run)[1].get("median")


AUTH_LIMITATION = "Comps aren't verified authentic; buyers expect authentication (e.g. eBay Authenticity Guarantee)."
NEW_BELOW_RETAIL = 0.75  # sealed new on eBay at or under this share of retail is a key factor


def retail_status(run: Run) -> dict:
    """Why there is or isn't a retail price, in one code-written note the Writer and page both use.
    status: found | successor | boutique_only (Chanel, Hermès, Rolex: no online first-party retail)
    | discontinued (only resale results) | not_found."""
    ref = retail_reference(run)
    boutique = item_profile(run.item)["boutique_brand"]
    if ref and not ref["successor"]:
        return {"status": "found", "note": ""}
    if ref:
        return {"status": "successor", "note": "No new retail price for this exact model, so it is likely discontinued: the ceiling is its successor model's price, a looser one."}
    if boutique:
        return {"status": "boutique_only", "note": f"Not sold online: {boutique} sells new only in its boutiques, so there is no online retail price. The price comes from resale comps alone."}
    if no_new_retail(run):
        return {"status": "discontinued", "note": "No new retail found: Google Shopping returned only resale listings, so the item is likely discontinued. The price comes from comps alone."}
    retail = [f for f in run.findings if f["researcher"] == "retail"]
    if any(f.get("no_mainstream") for f in retail):
        return {"status": "no_mainstream", "note": "No mainstream retail price: no big-box or brand store lists it new online, only third-party sellers, "
                                                   "whose prices aren't used as a ceiling. The price comes from comps alone."}
    if retail and all(f["errors"] for f in retail):
        # every retail search errored: say why, don't present it as "not sold" or "discontinued"
        reason = api_reason(next(iter(retail[-1]["errors"].values())))
        return {"status": "search_failed", "reason": reason,
                "note": f"The retail price couldn't be checked: {reason.rstrip('.')}. The price comes from comps alone."}
    return {"status": "not_found", "note": "No current new retail price found online. The price comes from comps alone."}


def new_vs_retail(run: Run) -> dict | None:
    """Sealed-new resale far below the retail price (AirPods: $130 new on eBay vs $250 at Best
    Buy) is a key factor: buyers can get new for less than retail, which drags used prices down."""
    ref = retail_reference(run)
    if not ref or ref["successor"]:
        return None
    if run.item.condition == "new":
        ids = kept_comp_ids(run)
    else:
        tiers = next((f for f in run.findings if f["researcher"] == "condition_tiers" and f.get("tier_ids", {}).get("new")), None)
        ids = tiers["tier_ids"]["new"] if tiers else []
    stats = market.comp_stats([run.sources[s] for s in ids])
    basis = "price_with_shipping" if stats["price_with_shipping"]["n"] * 2 >= stats["item_price"]["n"] > 0 else "item_price"
    st = stats[basis]
    if st.get("n", 0) < 3 or not ref["price"]:
        return None
    ratio = st["median"] / ref["price"]
    if ratio > NEW_BELOW_RETAIL:
        return None
    sentence = (f"Sealed new listings on eBay ask a median {_money(st['median'])} across {st['n']} sealed new listings, "
                f"{round((1 - ratio) * 100)}% below retail ({ceiling_text(ref)}): "
                f"buyers can get it new for well under retail, which pulls used prices down.")
    return {"n": st["n"], "median": st["median"], "retail_price": ref["price"], "ratio": round(ratio, 2), "sentence": sentence}


def comp_summary(run: Run) -> dict:
    """The exact comps across every round, with one count: the kept comps."""
    comps = comps_findings(run)
    basis, stats = comp_basis(run)
    funnel = comp_funnel(run)
    return {
        "kept": funnel["kept"],
        "found": funnel["found"],
        "after_title_filter": funnel["after_title_filter"],
        "excluded_summary": funnel["excluded_summary"],
        "exclusion_sentence": exclusion_sentence(funnel),
        "broadened": any(f["broadened"] for f in comps),
        "queries": [q for f in comps for q in f["queries"]],
        "price_basis": basis,
        "stats_n": stats.get("n", 0),
        "stats": stats,
        "median_phrase": comp_median(run)["phrase"],
        "grade": grade_summary(run),
        "anchor_median": anchor_median(run),
        "best_offer_count": best_offer_count(run),
        "best_offer_share": best_offer_share(run),
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


def fix_counts(memo: "Memo", kept: int) -> "Memo":
    """The last resort when the Writer keeps a wrong comp count. In the headline and the confidence
    reason (about the comps by design) the number becomes the kept count; elsewhere the sentence
    is dropped, since a wrong count may stand for some other subset and guessing which is worse."""
    def wrong_at(text: str) -> list:
        return [m for m in COUNT_CLAIM.finditer(text) if int(m.group(1)) != kept
                and not re.search(r"\b(excluded|excluding|dropped|removed|left out)\b", text[max(0, m.start() - 25):m.start()].lower())]

    def renumber(text: str) -> str:
        for m in reversed(wrong_at(text)):
            text = text[:m.start(1)] + str(kept) + text[m.end(1):]
        return text

    def drop(text: str) -> str:
        return " ".join(s for s in SENTENCE.split(text) if not wrong_at(s))

    return memo.model_copy(update={
        "headline": renumber(memo.headline),
        "confidence_reason": renumber(memo.confidence_reason),
        "rationale": drop(memo.rationale),
        "key_factors": [k for k in memo.key_factors if not wrong_at(k)],
    })


def comp_spread(run: Run) -> float | None:
    """IQR / median of the kept comps: 0.2 means the middle half sits within 20% of the median."""
    st = comp_basis(run)[1]
    return round(st["iqr"] / st["median"], 3) if st.get("median") else None


def no_new_retail(run: Run) -> bool:
    return any(f["researcher"] == "retail" and f["only_resale_listings"] for f in run.findings)


# Stores whose new price is a real retail price. With two or more of their offers, the ceiling
# comes from them alone; a marketplace third-party seller's offer can't set it.
MAJOR_RETAILERS = ("best buy", "target", "walmart", "amazon", "apple", "costco", "sam's club", "macy's", "nordstrom",
                   "kohl's", "dick's", "foot locker", "finish line", "jd sports", "champs", "zappos", "academy",
                   "b&h", "newegg", "staples", "home depot", "lowe's", "gamestop", "verizon", "at&t", "t-mobile",
                   "bloomingdale", "saks", "neiman marcus", "dsw", "rei", "ulta", "sephora", "kicks crew",
                   "playstation direct", "microsoft", "samsung", "google store", "dell", "lenovo", "micro center",
                   "adorama", "office depot", "bj's", "meijer", "kroger", "jcpenney", "belk", "dillard", "scheels",
                   "bass pro", "cabela", "wayfair", "bed bath", "qvc", "hsn", "nordstrom rack", "anthropologie",
                   "urban outfitters", "gap", "old navy", "h&m", "uniqlo", "zara", "nike", "adidas", "dyson",
                   "nintendo", "sony", "lululemon")
OUTLIER_SHARE = 0.6  # an offer under this share of the retail median is a deal or a mistake, not the ceiling


def is_major_retailer(retailer: str, product: str) -> bool:
    """A big-box store, or the brand's own store ("Nike" for a Nike shoe)."""
    r = (retailer or "").lower()
    brand = (product or "").split()[0].lower() if product else ""
    first_party = len(brand) > 2 and r.split(" - ")[0].strip().startswith(brand)
    # "Walmart - Cellfeee" is a marketplace seller on Walmart, not Walmart
    return first_party or (" - " not in r and any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", r) for m in MAJOR_RETAILERS))


def retail_reference(run: Run) -> dict | None:
    """The retail ceiling and how resale compares to it. The item's own retail offers win over its
    successor's. Robust to one cheap offer: the ceiling is the median of major-retailer offers
    (all kept offers when fewer than two are major), after dropping offers under OUTLIER_SHARE of
    that median. The cheapest offer is reported as a deal (lowest_note), not as the ceiling.
    role: "ceiling" if comps sit below it, "premium" if above."""
    retail = [f for f in run.findings if f["researcher"] == "retail"]
    for successor in (False, True):
        ids = [sid for f in retail if f["successor"] == successor for sid in f["kept_ids"]]
        priced = [sid for sid in dict.fromkeys(ids) if run.sources[sid]["extracted_price"] is not None]
        if any(is_major_retailer(run.sources[s]["retailer"], run.item.product) for s in priced):
            break
    else:
        return None
    p = lambda s: run.sources[s]["extracted_price"]  # noqa: E731
    # one offer per retailer, its cheapest: ten farfetch colorways are one store's price, not ten votes
    per_store: dict[str, str] = {}
    for s in sorted(priced, key=p):
        per_store.setdefault((run.sources[s]["retailer"] or s).lower(), s)
    stores = list(per_store.values())
    # Only big-box and brand stores set the ceiling: a marketplace seller's markup isn't a retail price
    pool = major = [s for s in stores if is_major_retailer(run.sources[s]["retailer"], run.item.product)]
    if not pool:
        return None
    first = market.price_stats([p(s) for s in pool])["median"]
    outliers = [s for s in pool if p(s) < OUTLIER_SHARE * first]
    pool = [s for s in pool if s not in outliers] or pool
    ceiling = market.price_stats([p(s) for s in pool])["median"]
    single = pool[0] if len(pool) == 1 else None
    if single:
        basis, retailer = f"{run.sources[single]['retailer']}'s price", run.sources[single]["retailer"]
    else:
        names = list(dict.fromkeys(run.sources[s]["retailer"] for s in sorted(pool, key=p)))
        basis = f"median of {len(pool)} major retailers: {', '.join(names[:3])}{', …' if len(names) > 3 else ''}"
        retailer = "major retailers"
    lowest = min(priced, key=p)
    lowest_note = None
    if p(lowest) < 0.95 * ceiling and lowest != single:
        r = run.sources[lowest]
        how = f"has it on sale for {_money(p(lowest))} (was {_money(r['extracted_old_price'])})" if r.get("extracted_old_price") else f"lists it for {_money(p(lowest))}"
        lowest_note = f"One retailer, {r['retailer']}, {how} [{lowest}]: a deal, not the ceiling."
    median = comp_basis(run)[1].get("median")
    ratio = round(median / ceiling, 2) if median and ceiling else None
    return {
        "id": single,  # a single offer's id when one offer is the ceiling; else the ceiling is a median
        "price": ceiling,
        "basis": basis,
        "retailer": retailer,
        "pool_ids": pool,
        "outlier_ids": outliers,
        "lowest": {"id": lowest, "price": p(lowest), "retailer": run.sources[lowest]["retailer"]},
        "lowest_note": lowest_note,
        "successor": successor,
        "median_new": market.price_stats([p(s) for s in priced])["median"],
        "on_sale_ids": [s for s in priced if run.sources[s]["old_price"]],
        "comp_median": median,
        "resale_to_retail_ratio": ratio,
        "role": "premium" if ratio is not None and ratio > 1 else "ceiling",
    }


def ceiling_text(ref: dict) -> str:
    """'$110.00 (median of 4 major retailers: ...)' or '$249.95 at Best Buy [R1]'."""
    if ref["id"]:
        return f"{_money(ref['price'])} at {ref['retailer']} [{ref['id']}]"
    return f"{_money(ref['price'])} ({ref['basis']})"


COLLECTIBLE = r"\bcollectible\b|\blimited\b|\brare\b|\bsigned\b|\bautographed\b|\bfirst edition\b|\bretro\b|\bdiscontinued\b|\bgraded\b"


def mixed_lines(run: Run) -> list[str]:
    """Product lines the kept comps name when the seller named none ('Dyson V8' comps mixing
    Animal, Absolute and Origin): two or more means the comps aren't one product."""
    if re.search(rf"\b({PRODUCT_LINES})\b", run.item.product or "", re.I):
        return []
    found = Counter(m.lower() for s in kept_comp_ids(run) for m in set(re.findall(rf"\b({PRODUCT_LINES})\b", run.sources[s]["title"] or "", re.I)))
    return sorted(found) if len(found) >= 2 else []


def above_retail(run: Run) -> bool:
    """A non-new everyday item whose comps ask more than the retail ceiling: suspicious. Vintage,
    collectible and luxury items can resell above retail, so they're exempt."""
    if run.item.condition == "new":
        return False
    prof = item_profile(run.item)
    if prof["luxury"] or prof["similar_items"] or re.search(COLLECTIBLE, _item_text(run.item), re.I):
        return False
    ref = retail_reference(run)
    return bool(ref and ref["role"] == "premium")


def confidence_cap(run: Run) -> tuple[str, str]:
    """The highest confidence the data supports, from the kept comp count and their price spread.
    The Writer can go lower, never higher. Missing retail doesn't lower it; a successor ceiling does."""
    n_all = len(kept_comp_ids(run))
    if item_profile(run.item)["similar_items"]:
        return "low", "the comps are similar items, not exact matches"
    n_exact = max((f["n_kept"] for f in comps_findings(run) if not f["broadened"]), default=0)
    spread = comp_spread(run)
    ref = retail_reference(run)
    if n_all < MIN_COMPS:
        return "low", f"only {n_all} comparable listings survived filtering"
    spread_text = f"the middle half of prices spans {spread:.0%} of the median" if spread is not None else ""
    if spread is not None and spread > WIDE_SPREAD:
        return "low", f"comp prices vary widely: {spread_text}"
    if above_retail(run):
        return "medium", "used listings ask more than the retail ceiling, which for an everyday item points to newer models, bundles or new units in the comps"
    if run.item.condition == "for_parts":
        return "medium", "for-parts prices depend on what's wrong, which listing titles rarely say"
    if lines := mixed_lines(run):
        return "medium", f"the comps mix product lines that price differently ({', '.join(lines)}); naming the exact model would tighten the range"
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
    findings = [{k: v for k, v in f.items() if k not in ("excluded", "kept_ids", "tier_ids")} for f in run.findings if f["researcher"] != "exact_comps"]
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
        "retail_status": retail_status(run),
        "new_vs_retail": new_vs_retail(run),
        "item_profile": item_profile(run.item),
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


SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
# A sentence that states exclusion counts: "Excluded 4 listings", "3 single earbuds were removed"
_COUNT = r"(?<![$\d.,])\b\d+\b(?![%\d.,]|\s*(?:x|×)\b)"  # a bare count: not $80, 20%, 1.4x
_EXCLUDED = r"\b(?:exclud\w*|filtered out|dropped|removed|left out)\b"
EXCLUSION_COUNT = re.compile(rf"{_COUNT}.*{_EXCLUDED}|{_EXCLUDED}.*{_COUNT}", re.I)


def apply_code_text(run: Run, clean: dict, warnings: list[str]) -> None:
    """Put the sentences code owns into the memo, in place of any the Writer wrote itself: the
    exclusion breakdown (same data as the chips), the grade and new-vs-retail key factors, and the
    authenticity limitation for luxury items."""
    sentence = exclusion_sentence(comp_funnel(run))
    parts = [s for s in SENTENCE.split(clean["rationale"].strip()) if s]
    own = [s for s in parts if EXCLUSION_COUNT.search(s)]
    if own and not all(s.strip() == sentence for s in own):
        warnings.append("the Writer's exclusion counts were replaced with the code-generated breakdown")
    clean["rationale"] = " ".join([s for s in parts if not EXCLUSION_COUNT.search(s)] + [sentence])
    clean["key_factors"] = [k for k in clean["key_factors"] if not EXCLUSION_COUNT.search(k)]
    clean["limitations"] = [x for x in clean["limitations"] if not EXCLUSION_COUNT.search(x)]
    # Best Offer only when a meaningful share of comps take offers
    if best_offer_share(run) < BEST_OFFER_SHARE:
        clean["key_factors"] = [k for k in clean["key_factors"] if not BEST_OFFER.search(k)]
        clean["rationale"] = " ".join(s for s in SENTENCE.split(clean["rationale"]) if not BEST_OFFER.search(s))
    # Code's factors first; a Writer factor on the same topic, or a near-copy of another, is dropped
    gap, grade = new_vs_retail(run), grade_summary(run)
    added = [gap["sentence"]] if gap else []
    writer = [k for k in clean["key_factors"] if not (gap and NEW_BELOW_RETAIL_TOPIC.search(k))]
    if grade and grade["note"]:
        added.append(grade["note"])
        writer = [k for k in writer if not GRADE_TOPIC.search(k)]
    clean["key_factors"] = dedupe_factors(added + writer)
    if item_profile(run.item)["luxury"]:
        clean["limitations"] = [x for x in clean["limitations"] if "authentic" not in x.lower()] + [AUTH_LIMITATION]
    if above_retail(run):
        ref = retail_reference(run)
        clean["limitations"] = [f"Used listings ask more than the retail ceiling ({_money(ref['comp_median'])} vs {ceiling_text(ref)}): for an "
                                "everyday item that usually means the comps include newer models, bundles or new units, so check them "
                                "before listing."] + clean["limitations"]
    if item_profile(run.item)["similar_items"]:
        clean["rationale"] = f"{SIMILAR_NOTE} {clean['rationale']}"
        clean["limitations"] = [SIMILAR_NOTE.split(":")[0] + "; the range is a rough guide."] + clean["limitations"]
    for k in ("headline", "rationale", "confidence_reason"):
        clean[k] = fix_stat_citations(clean[k], run)
    for k in ("key_factors", "limitations"):
        clean[k] = [fix_stat_citations(x, run) for x in clean[k]]


BEST_OFFER = re.compile(r"\bbest[- ]offers?\b|\baccepts?(ing)? offers\b|\bmake an offer\b|\bopen to offers\b", re.I)
NEW_BELOW_RETAIL_TOPIC = re.compile(r"\bsealed\b|\bnew\b.*\b(below|under|less than|cheaper than)\b.*\bretail\b|\bbelow retail\b", re.I)
GRADE_TOPIC = re.compile(r"\bpre-owned - \w+|\b(your|matching|same) grade\b|\bgrade[- ](weighted|median)\b", re.I)


def _words(text: str) -> set[str]:
    return {w.rstrip(".,") for w in re.findall(r"\$?\d[\d,.]*|[a-z]+", TAG.sub("", text).lower())} - {"the", "a", "an", "of", "and", "to", "at", "on", "in", "for", "is"}


def dedupe_factors(factors: list[str], overlap: float = 0.6) -> list[str]:
    """Drop a key factor that repeats an earlier one (the same words, give or take a few)."""
    out = []
    for f in factors:
        w = _words(f)
        if f.strip() and not any(w and len(w & _words(o)) / len(w | _words(o)) >= overlap for o in out):
            out.append(f)
    return out


AMOUNT = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
TAG_RUN = re.compile(r"(?:\s*\[[ER]\d+\])+")
STAT_WORDS = re.compile(r"\b(median|q1|q3|quartiles?|average|mean|weighted|anchor|middle half|typical)\b", re.I)
BASIS_NAMED = re.compile(r"\b(comps|listings|offers|retailers|stores)\b", re.I)


def _same_amount(written: str, value) -> bool:
    """'$75' is 75.00 (and rounds 74.99); '$75.25' must match to the cent."""
    if not isinstance(value, (int, float)):  # retail "price" is display text ("$249.95"); extracted_price is the number
        return False
    x = float(written.replace(",", ""))
    return abs(x - value) < 0.005 or ("." not in written and abs(x - value) < 0.5)


def stat_labels(run: Run) -> list[tuple[float, str, str]]:
    """(value, kind, basis label) for every statistic the memo may quote."""
    out = []
    cm = comp_median(run)
    over = cm["phrase"].split(" across ", 1)[1] if " across " in cm["phrase"] else f"{cm['of']} comps"
    for kind, name in (("median", "median"), ("q1", "Q1"), ("q3", "Q3")):
        if cm.get(kind) is not None:
            out.append((cm[kind], kind, f"{name} of {over.removeprefix('the ')}"))
    if g := grade_summary(run):
        if g["weighted_median"] is not None:
            out.append((g["weighted_median"], "weighted", f"grade-weighted median of {g['n']} comps"))
        if g["matching_median"] is not None:
            out.append((g["matching_median"], "grade", f"median of the {g['n_matching']} {g['label']} comps"))
    for t in tier_rows(run):
        if t["source"] == "condition_search" and t["median"] is not None:
            label = CONDITION_LABELS.get(t["condition"], t["condition"]).lower()
            out.append((t["median"], "median", f"median of {t['n']} {label} listings in the broader condition search"))
    if ref := retail_reference(run):
        if not ref["id"]:  # a ceiling that is a median is cited to its basis
            out.append((ref["price"], "median", ref["basis"]))
        if ref.get("median_new") is not None:
            out.append((ref["median_new"], "median", "median of the new retail offers"))
    return out


def _pick_label(written: str, context: str, labels: list) -> str | None:
    found = [x for x in labels if _same_amount(written, x[0])]
    if not found:
        return None
    for kind, pattern in (("weighted", r"\bweighted\b|\banchor\b"), ("q1", r"\bq1\b|\blower quartile\b"), ("q3", r"\bq3\b|\bupper quartile\b"),
                          ("grade", r"\bgrade\b|\bpre-owned\b")):
        if re.search(pattern, context, re.I):
            return next((x[2] for x in found if x[1] == kind), found[0][2])
    return found[0][2]


def fix_stat_citations(text: str, run: Run) -> str:
    """'$7,036.00 [E3]' where $7,036 is a weighted median -> '$7,036.00 (grade-weighted median of 16
    comps)'. A tag after a dollar amount stays only when the amount is that listing's own price,
    total or shipping and isn't described as a statistic; a statistic is cited to its basis."""
    labels = stat_labels(run)
    edits, seg_start = [], 0

    def own_price(i: str, written: str) -> bool:
        return i in run.sources and any(_same_amount(written, run.sources[i].get(k))
                                        for k in ("price", "totalPrice", "shippingCost", "extracted_price", "extracted_old_price"))

    for m in TAG_RUN.finditer(text):
        ids = re.findall(r"[ER]\d+", m.group(0))
        # a tag that names the listing before its price: "[E34] at $75.95", "[R2] for $99"
        lead = re.match(r"\s*(?:,|:|\(|at|for|listed at|asks?|asking)?\s*\$\s?(\d[\d,]*(?:\.\d{1,2})?)", text[m.end():])
        if lead:
            keep = [i for i in ids if own_price(i, lead.group(1))]
            if keep != ids:
                edits.append((m.start(), m.end(), "".join(f"[{i}]" for i in keep) and " " + "".join(f"[{i}]" for i in keep)))
            seg_start = m.end() + lead.end()  # that amount is claimed; a later tag can't pair with it
            continue
        boundary = max((b.end() for b in re.finditer(r"[.!?;:]\s", text[:m.start()])), default=0)
        start = max(seg_start, boundary)
        seg_start = m.end()
        amounts = list(AMOUNT.finditer(text, start, m.start()))
        if not amounts:
            continue
        a = amounts[-1]
        before, between = text[max(start, a.start() - 40):a.start()], text[a.end():m.start()]
        is_stat = bool(STAT_WORDS.search(before) or STAT_WORDS.search(between))
        keep = [] if is_stat else [i for i in ids if own_price(i, a.group(1))]
        if keep == ids:
            continue
        edits.append((m.start(), m.end(), "".join(f"[{i}]" for i in keep) and " " + "".join(f"[{i}]" for i in keep)))
        after = re.match(r"\s*\([^)]*\)", text[m.end():])  # "$94.50 [R18] (median of 2 retailers)"
        if not keep and not BASIS_NAMED.search(before + between + (after.group(0) if after else "")):
            if label := _pick_label(a.group(1), before + between, labels):
                edits.append((a.end(), a.end(), f" ({label})"))
    for s, e, rep in sorted(edits, key=lambda x: x[0], reverse=True):
        text = text[:s] + rep + text[e:]
    return text


def split_assumptions(assumptions: list[str]) -> list[str]:
    """One detail per bullet: authenticity is a limitation, never an assumption."""
    out = []
    for a in assumptions:
        out += [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", a or "") if s.strip()]
    return [a for a in dict.fromkeys(out) if not re.search(r"authentic", a, re.I)]


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
    writer_texts = [clean["headline"], clean["rationale"], clean["confidence_reason"], *clean["key_factors"]]
    apply_code_text(run, clean, warnings)
    # a tag on a listing no researcher kept points the reader at something the price doesn't use
    used = {i for f in run.findings for i in f["kept_ids"]}
    dropped = {e["id"] for f in run.findings for e in f["excluded"]} - used
    stripped = set()

    def strip_excluded(text: str) -> str:
        def sub(m):
            stripped.add(m.group(1))
            return ""
        return re.sub(r"\s*\[(" + "|".join(map(re.escape, sorted(dropped))) + r")\]", sub, text) if dropped else text

    for k in ("headline", "rationale", "confidence_reason"):
        clean[k] = strip_excluded(clean[k])
    for k in ("key_factors", "limitations"):
        clean[k] = [strip_excluded(x) for x in clean[k]]
    if stripped:
        warnings.append(f"removed citations to excluded listings: {', '.join(sorted(stripped, key=lambda s: (s[0], int(s[1:]))))}")
    # what's cited after code's edits (stat tags dropped, code sentences added)
    final = [clean["headline"], clean["rationale"], clean["confidence_reason"], *clean["key_factors"], *clean["limitations"]]
    cited = [i for t in final for i in re.findall(r"\[([ER]\d+)\]", t) if i in run.sources]
    excluded = {e["id"] for f in run.findings for e in f["excluded"]}
    if wrong := sorted(set(cited) & excluded - set(kept_comp_ids(run)), key=lambda s: int(s[1:])):
        warnings.append(f"cites listings that were excluded from the comps: {', '.join(wrong)}")
    low, high = sorted((round(memo.price_low, 2), round(memo.price_high, 2)))
    ref = retail_reference(run)
    if ref and ref["role"] == "ceiling" and high > ref["price"]:
        warnings.append(f"the range tops the retail ceiling ({ceiling_text(ref)})")
    # only the Writer's own sentences: code's ("16 sealed new listings") state other counts on purpose
    if wrong := count_mismatches(writer_texts, len(kept_comp_ids(run))):
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
    basis = comp_basis(run)[0]
    grade = seller_grade(run.item)
    for sid in sorted(kept_comp_ids(run), key=lambda s: run.sources[s]["price"]):
        r = run.sources[sid]
        g = listing_grade(r["condition"])
        rows.append({"id": sid, "title": r["title"], "condition": r["condition"], "price": r["price"],
                     "shipping": r["shippingCost"], "total": r["totalPrice"], "url": r["itemWebUrl"], "image": r.get("image"),
                     "best_offer": "BEST_OFFER" in r["buyingOptions"],
                     # in the median: on a price + shipping basis, a comp with unknown shipping isn't
                     "in_median": _basis_value(r, basis) is not None,
                     "grade": g, "grade_match": bool(grade) and g == grade,
                     "weight": grade_weight(g, grade) if grade else 1.0})
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
        cm = comp_median(run)
        # n is the listings the median is over; of is every kept comp ("19 of 27 list shipping")
        rows[run.item.condition] = {"condition": run.item.condition, "n": cm["n"], "of": cm["of"], "median": cm["median"],
                                    "q1": cm["q1"], "q3": cm["q3"], "basis": cm["basis"], "source": "exact_comps"}
    for f in run.findings:
        if f["researcher"] != "condition_tiers":
            continue
        for tier, s in (f.get("stats_by_condition") or {}).items():
            # the same basis rule as the exact comps
            basis = "price_with_shipping" if s["price_with_shipping"]["n"] * 2 >= s["item_price"]["n"] > 0 else "item_price"
            st = s[basis]
            if st["n"] and tier not in rows:
                rows[tier] = {"condition": tier, "n": st["n"], "of": s["item_price"]["n"], "median": st["median"], "q1": st["q1"], "q3": st["q3"],
                              "basis": basis, "source": "condition_search"}
    order = list(market.CONDITION_IDS)
    return sorted(rows.values(), key=lambda r: order.index(r["condition"]) if r["condition"] in order else len(order))


def retail_rows(run: Run) -> list[dict]:
    ids = list(dict.fromkeys(sid for f in run.findings if f["researcher"] == "retail" for sid in f["kept_ids"]))
    successor = {sid for f in run.findings if f["researcher"] == "retail" and f["successor"] for sid in f["kept_ids"]}
    rows = []
    for s in ids:
        r = run.sources[s]
        title, note = retail_title(r["title"], run.item.product, s in successor)
        rows.append({"id": s, "retailer": r["retailer"], "title": title, "listed_as": r["title"], "model_note": note,
                     "price": r["extracted_price"], "old_price": r["extracted_old_price"], "url": r["link"]})
    return sorted((r for r in rows if r["price"] is not None), key=lambda r: r["price"])[:8]


# Store-page words that say nothing about which product it is
OFFER_PREFIX = re.compile(r"^\s*(personali[sz]ed|engraved|customi[sz]able|custom engraving)\s+", re.I)


def retail_title(title: str, product: str, successor: bool = False) -> tuple[str, str]:
    """The clearest name for a retail offer, and a note naming the model when the store's title
    leaves it out. 'Personalized AirPods Pro' for an AirPods Pro 2 search -> ('AirPods Pro',
    'personalization option; store title doesn't state the model; searched for AirPods Pro 2')."""
    title = title or ""
    notes = []
    clean = OFFER_PREFIX.sub("", title)
    if clean != title:
        notes.append("personalization option")
    # model numbers and generations the product names but the title doesn't ("2", "2nd", "13")
    marks = set(re.findall(r"\b(\d+(?:st|nd|rd|th)?)\b", product or "", re.I))
    have = set(re.findall(r"\b(\d+(?:st|nd|rd|th)?)\b", clean, re.I))
    gen = lambda m: re.sub(r"(st|nd|rd|th)$", "", m.lower())  # noqa: E731  "2nd" and "2" are the same
    if not successor and {gen(m) for m in marks} - {gen(m) for m in have}:
        notes.append(f"store title doesn't state the model; searched for {product}")
    return clean.strip() or title, "; ".join(notes)


def render(run: Run, memo: dict, rows: dict, cited: list[str]) -> tuple[str, dict]:
    """The memo as markdown, plus the sources it cites. Every table number comes from code."""
    funnel = comp_funnel(run)
    n_found, n_title, n_kept = funnel["found"], funnel["after_title_filter"], funnel["kept"]
    table = spread(rows["comps"])
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
    ]
    if run.assumptions:
        lines += ["", "### Assumptions", *[f"- {a}" for a in run.assumptions]]
    lines += [
        "",
        f"### Comparable listings ({n_kept} kept)",
        f"{n_found} found, {n_title} after the title filter, {n_kept} after review. {CONDITION_LABELS.get(run.item.condition, run.item.condition)} {comp_median(run)['phrase']}."
        + (f" Showing {len(table)} spread across the range, cheapest first." if len(table) < n_kept else ""),
        "",
        "| Source | Listing | Condition | Price | Shipping | Total |",
        "|---|---|---|---|---|---|",
        *[f"| [{r['id']}] | {r['title'][:70]} | {r['condition']} | {_money(r['price'])} | {_money(r['shipping'])} | {_money(r['total'])} |" for r in table],
    ]
    if funnel["excluded"]:
        lines += ["", exclusion_sentence(funnel)]
    if rows["tiers"]:
        lines += ["", "### Price by condition (active listings)", "| Condition | Listings | Median | Middle half |", "|---|---|---|---|"]
        lines += [f"| {t['condition']} | {t['n'] if t['n'] == t['of'] else f'{t['n']} of {t['of']}'} | {_money(t['median'])} | {_money(t['q1'])} – {_money(t['q3'])} |" for t in rows["tiers"]]
    lines += ["", "### Retail (new)"]
    ref = retail_reference(run)
    if ref:
        model = "the successor model's " if ref["successor"] else ""
        x = ref["resale_to_retail_ratio"]
        if ref["role"] == "premium":
            lines += [f"Resale premium: comps ask {x:.2f}x {model}retail price ({_money(ref['comp_median'])} vs "
                      f"{ceiling_text(ref)}), so the price comes from comps alone.", ""]
        elif x is not None:
            lines += [f"Resale-to-retail ratio {x:.2f}: comps sit below {model}retail ceiling of {ceiling_text(ref)}.", ""]
        if ref.get("lowest_note"):
            lines += [ref["lowest_note"], ""]
    status = retail_status(run)
    if status["status"] == "successor":
        lines += [status["note"], ""]
    if rows["retail"]:
        lines += ["| Source | Retailer | Offer | Price | Was |", "|---|---|---|---|---|"]
        lines += [f"| [{r['id']}] | {r['retailer']} | {r['title'][:60]}{f' ({r['model_note']})' if r['model_note'] else ''} | {_money(r['price'])} | "
                  f"{_money(r['old_price']) if r['old_price'] else ''} |" for r in rows["retail"]]
    else:
        lines += [status["note"]]
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
        # eBay itself failed (quota, credentials): say that, not "couldn't find enough listings"
        failed = [api_reason(e) for f in comps_findings(run) for e in f["errors"].values()]
        if failed and not any(f["n_found"] for f in comps_findings(run)):
            return f"I couldn't search eBay, so I can't price this yet: {failed[0]}", None
        searched = "; ".join(f"{f['researcher']}: {', '.join(f['queries'])}" for f in run.findings) or "nothing"
        retail = retail_reference(run)
        extra = f" For reference, it sells new for {ceiling_text(retail)}." if retail else ""
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
        # Still off after the retry: code corrects it, so the memo never states a wrong comp count
        if still := count_mismatches([draft.headline, draft.rationale, draft.confidence_reason, *draft.key_factors], kept):
            draft = fix_counts(draft, kept)
            run.trace.record("writer", "recount (code fix)", {"counts_found": sorted(set(still)), "kept": kept},
                             {"counts_left": count_mismatches([draft.headline, draft.rationale, draft.confidence_reason, *draft.key_factors], kept)})
    memo, check = validate_memo(run, draft, cap)
    rows = {"comps": comp_rows(run), "tiers": tier_rows(run), "retail": retail_rows(run)}
    markdown, sources = render(run, memo, rows, check["cited"])
    run.trace.record("writer", "validate_citations", {"tags_in_memo": check["cited"] + check["unknown_tags_stripped"]}, check)
    funnel = comp_funnel(run)
    return markdown, {
        "item": run.item.model_dump(),
        "item_profile": item_profile(run.item),
        "assumptions": run.assumptions,
        **memo,
        "comps": rows["comps"],
        "comps_found": funnel["found"],
        "comps_after_title_filter": funnel["after_title_filter"],
        "comps_kept": funnel["kept"],
        "comps_excluded": funnel["excluded"],
        "excluded_summary": funnel["excluded_summary"],
        "excluded_groups": funnel["excluded_groups"],
        "exclusion_sentence": exclusion_sentence(funnel),
        "comp_median": comp_median(run),
        "grade": grade_summary(run),
        "new_vs_retail": new_vs_retail(run),
        "condition_tiers": rows["tiers"],
        "retail": rows["retail"],
        "retail_reference": retail_reference(run),
        "retail_status": retail_status(run),
        "no_new_retail_found": no_new_retail(run) and retail_reference(run) is None,
        "comp_price_spread": comp_spread(run),
        "sources": sources,
        "citation_check": check,
        "rounds": run.rounds,
    }


# --- The whole pipeline ---


WEAPON = re.compile(r"\b(rifles?|shotguns?|handguns?|pistols?|revolvers?|firearms?|guns?|ammo|ammunition|ar-?15s?|ak-?47s?"
                    r"|suppressors?|silencers?|muzzle brakes?|bump stocks?|gun magazines?|crossbows?|switchblades?|brass knuckles)\b", re.I)
NOT_A_WEAPON = re.compile(r"\b(glue|heat|nail|staple|spray|massage|nerf|water|toy|caulk|caulking|grease|paint|airbrush|label|price|"
                          r"temperature|radar|squirt|bubble|tattoo|solder(ing)?|top gun|pop|cap|lego)\s+guns?\b|\bguns? n'? roses\b", re.I)
WEAPON_REPLY = ("Sorry, I can't help price firearms, ammunition or other weapons: eBay doesn't allow them, and I don't price them. "
                "I'm happy to suggest an asking price for another secondhand item.")


def is_weapon(text: str) -> bool:
    """Firearms and weapons get a polite decline in code, before any research. A glue gun or a
    massage gun is not one."""
    return bool(WEAPON.search(NOT_A_WEAPON.sub(" ", text or "")))


def transcript(turns: list[dict]) -> str:
    return "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)


async def answer(turns: list[dict], clarifications: int) -> dict:
    """turns: the session's {role, content} messages, the latest user message last.
    Returns {response, tool_calls, memo, clarifying}."""
    trace = Trace()
    if is_weapon(turns[-1]["content"]):
        trace.record("lead", "decline", {"message": turns[-1]["content"]}, {"reason": "firearm or weapon"})
        return {"response": WEAPON_REPLY, "tool_calls": trace.calls, "memo": None, "clarifying": False}
    prompt = f"{transcript(turns)}\n\n(Clarifying questions already asked: {clarifications})"
    intake = (await Runner.run(intake_agent, prompt, hooks=trace, max_turns=2)).final_output
    trace.record("lead", "assess_item", {"conversation": transcript(turns)}, intake.model_dump())
    if not intake.ready or intake.item is None:
        question = intake.question or "What item is it (brand and model), and what condition is it in?"
        return {"response": question, "tool_calls": trace.calls, "memo": None, "clarifying": True}

    item = intake.item
    user_text = turns[-1]["content"]  # the latest message; earlier ones may be about another item
    if (mapped := map_condition(item, user_text)) != item.condition:
        trace.record("lead", "map_condition", {"intake": item.condition, "words": item.condition_detail}, {"condition": mapped})
        item = item.model_copy(update={"condition": mapped})
    # a variant detail may come from an answer to a clarifying question: check every user turn
    said = " ".join(t["content"] for t in turns if t["role"] == "user")
    if (unguessed := drop_guessed_variant(item, said)) != item:
        trace.record("lead", "drop_guessed_variant", {"variant": item.variant}, {"variant": unguessed.variant, "search_query": unguessed.search_query})
        item = unguessed
    run = Run(item=item, assumptions=split_assumptions(intake.assumptions), trace=trace)
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


def offers_common(memo: dict) -> bool:
    comps = memo.get("comps") or []
    return bool(comps) and sum(bool(c.get("best_offer")) for c in comps) / len(comps) >= BEST_OFFER_SHARE


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
        "price_note": "The top of the suggested range" + (", leaving room for Best Offer." if offers_common(memo) else "."),
        "source": source,
    }
