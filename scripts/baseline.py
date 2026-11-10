"""Baseline for the README: the Lead's model with no tools and no system prompt.

    uv run python scripts/baseline.py

Sends each sample query to the same model the Lead uses (vertex_ai/gemini-3.5-flash), through
LiteLLM, as a single user message. Prints each answer.
"""

import os
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "vertex_ai/gemini-3.5-flash"  # same as research.MODEL
QUERIES = [
    "Used AirPods Pro 2, good condition, with charging case",
    "iPhone 13 Pro 128GB unlocked, used with light scratches",
    "Chanel Classic Flap medium, black caviar leather, gold hardware, used in good condition",
]

# Same Vertex settings as research.py: Gemini 3.5 is served only from the global location
VERTEX = {"vertex_location": os.environ.get("VERTEXAI_LOCATION") or "global"}
if os.environ.get("VERTEXAI_PROJECT"):
    VERTEX["vertex_project"] = os.environ["VERTEXAI_PROJECT"]


def ask(query: str) -> str:
    response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": query}], timeout=120, num_retries=2, **VERTEX)
    return response.choices[0].message.content or ""


def main() -> None:
    for query in QUERIES:
        try:
            text = ask(query)
        except Exception as e:
            sys.exit(f"Baseline failed on {query!r}: {type(e).__name__}: {str(e)[:300]}")
        print(f"=== {query}\n{text}\n")


if __name__ == "__main__":
    main()
