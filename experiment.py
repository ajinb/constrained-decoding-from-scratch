#!/usr/bin/env python3
"""Free-running vs prompted vs constrained JSON extraction on a real small model.

Needs torch and transformers. Writes results/experiment.json.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from automaton import compile_schema
from mask import Mask

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SERVICES = ["checkout", "payments", "search", "auth", "catalog", "ledger"]
SEVERITIES = ["low", "medium", "high", "critical"]
SCHEMA = {
    "type": "object",
    "properties": {
        "service": {"enum": SERVICES},
        "severity": {"enum": SEVERITIES},
        "minutes": {"type": "integer"},
    },
}
TEMPLATES = [
    "The {service} service had a {severity} severity incident that lasted {minutes} minutes.",
    "Incident report: {service} was impacted for {minutes} minutes. Severity was {severity}.",
    "For {minutes} minutes the {service} team fought a {severity} outage.",
    "{severity} severity. {service} degraded. Duration: {minutes} minutes.",
]
INSTRUCTION = (
    "Extract the service, severity and duration in minutes from the text. Reply with only JSON "
    'of the form {"service":"<name>","severity":"<level>","minutes":<integer>}.\n\nText: '
)
MAX_NEW = 48
SERVICES_KEYS = ("service", "severity", "minutes")


def cases(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(count):
        truth = {
            "service": rng.choice(SERVICES),
            "severity": rng.choice(SEVERITIES),
            "minutes": rng.randint(2, 180),
        }
        out.append({"text": TEMPLATES[i % len(TEMPLATES)].format(**truth), "truth": truth})
    return out


def parse(text: str) -> dict | None:
    """The document, if it is JSON that satisfies the schema. Whitespace and key order are fine."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    ok = (
        isinstance(parsed, dict)
        and set(parsed) == set(SCHEMA["properties"])
        and parsed["service"] in SERVICES
        and parsed["severity"] in SEVERITIES
        and type(parsed["minutes"]) is int
    )
    return parsed if ok else None


def score(text: str, truth: dict) -> dict:
    """valid: fits the schema. right: per field, judged on anything that parses as JSON at all,
    so a casing slip in one field does not count against the others."""
    try:
        loose = json.loads(text)
    except ValueError:
        loose = {}
    loose = loose if isinstance(loose, dict) else {}
    right = {k: loose.get(k) == truth[k] for k in truth}
    return {"valid": parse(text) is not None, "correct": all(right.values()), "right": right}


@torch.inference_mode()
def generate(model, tokenizer, prompt_ids, mask=None, nfa=None, max_new=MAX_NEW):
    states = nfa.initial() if nfa else None
    tokens, past, inputs, masking_ms = [], None, torch.tensor([prompt_ids]), 0.0
    for _ in range(max_new):
        out = model(input_ids=inputs, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        if nfa is None:
            token = int(logits.argmax())
            if token == tokenizer.eos_token_id:
                break
        else:
            started = time.perf_counter()
            options = mask.allowed(states)
            if not options:
                break
            index = torch.tensor(sorted(options))
            token = int(index[int(logits[index].argmax())])
            states = options[token]
            masking_ms += (time.perf_counter() - started) * 1000
        tokens.append(token)
        inputs = torch.tensor([[token]])
    return tokens, masking_ms


def prompt_ids(tokenizer, content: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


def run(model, tokenizer, data, label, mask=None, nfa=None, max_new=MAX_NEW):
    rows = []
    masking, tokens_total = 0.0, 0
    for case in data:
        ids = prompt_ids(tokenizer, INSTRUCTION + case["text"])
        tokens, ms = generate(model, tokenizer, ids, mask, nfa, max_new)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        rows.append({**score(text, case["truth"]), "text": text})
        masking += ms
        tokens_total += len(tokens)
    return {
        "mode": label,
        "cases": len(rows),
        "max_new_tokens": max_new,
        "valid_rate": round(statistics.mean(r["valid"] for r in rows), 3),
        "all_fields_correct_rate": round(statistics.mean(r["correct"] for r in rows), 3),
        "field_correct_rate": {
            k: round(statistics.mean(r["right"][k] for r in rows), 3) for k in SERVICES_KEYS
        },
        "masking_ms_per_token": round(masking / max(tokens_total, 1), 3),
        "examples": [r["text"] for r in rows[:3]],
    }


def state_growth() -> list[dict]:
    """Why providers reject maxItems and maxLength: bounded repetition copies states."""
    rows = []
    for limit in (5, 10, 20, 40):
        item = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "maxLength": limit},
            },
        }
        flat = compile_schema({"type": "array", "items": item})
        bounded = compile_schema({"type": "array", "items": item, "maxItems": limit})
        nested = compile_schema(
            {
                "type": "array",
                "items": {"type": "array", "items": item, "maxItems": limit},
                "maxItems": limit,
            }
        )
        rows.append(
            {
                "limit": limit,
                "unbounded_array_states": len(flat.edges),
                "bounded_array_states": len(bounded.edges),
                "nested_bounded_states": len(nested.edges),
            }
        )
    return rows


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    vocabulary = {i: tokenizer.decode([i]) for i in range(len(tokenizer))}
    # shortcut: tokens that decode to partial UTF-8 are excluded, so non-ASCII strings cannot appear
    vocabulary = {i: t for i, t in vocabulary.items() if "�" not in t}
    nfa = compile_schema(SCHEMA)
    started = time.perf_counter()
    mask = Mask(nfa, vocabulary)
    mask.allowed(nfa.initial())
    setup_ms = (time.perf_counter() - started) * 1000
    nfa_ws = compile_schema(SCHEMA, whitespace=True)
    mask_ws = Mask(nfa_ws, vocabulary)
    data = cases(200, 20261030)
    results = [run(model, tokenizer, data, "prompt only")]
    print(results[-1], flush=True)
    results.append(run(model, tokenizer, data, "constrained, compact grammar", mask, nfa))
    print(results[-1], flush=True)
    results.append(run(model, tokenizer, data, "constrained, whitespace allowed", mask_ws, nfa_ws))
    print(results[-1], flush=True)
    results.append(run(model, tokenizer, data, "constrained, cut off at 8 tokens", mask, nfa, 8))
    print(results[-1], flush=True)
    out = {
        "model": MODEL,
        "vocabulary": len(vocabulary),
        "mask_setup_ms": round(setup_ms, 1),
        "results": results,
        "state_growth": state_growth(),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/experiment.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["state_growth"], indent=1))


if __name__ == "__main__":
    main()
