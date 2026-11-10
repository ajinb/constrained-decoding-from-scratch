import json
import random
import string

import pytest

from automaton import compile_schema
from mask import Mask

SCHEMA = {
    "type": "object",
    "properties": {
        "service": {"enum": ["checkout", "auth"]},
        "severity": {"enum": ["low", "high"]},
        "minutes": {"type": "integer"},
        "paged": {"type": "boolean"},
        "note": {"type": "string", "maxLength": 5},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}},
    },
}
VALID = '{"service":"auth","severity":"low","minutes":12,"paged":true,"note":"hi","tags":["a","b"]}'


def test_accepts_valid_document():
    assert compile_schema(SCHEMA).matches(VALID)


@pytest.mark.parametrize(
    "bad",
    [
        VALID.replace('"auth"', '"billing"'),
        VALID.replace("12", "012"),
        VALID.replace("true", "yes"),
        VALID.replace('"hi"', '"too long"'),
        VALID.replace('["a","b"]', '["a",]'),
        VALID[:-1],
        VALID + " ",
    ],
)
def test_rejects_invalid_documents(bad):
    assert not compile_schema(SCHEMA).matches(bad)


SPACED = (
    '{ "service": "auth", "severity": "low", "minutes": 12, "paged": true, '
    '"note": "hi", "tags": [ "a", "b" ] }'
)


def test_whitespace_grammar_accepts_spaced_json_and_compact_rejects_it():
    assert compile_schema(SCHEMA, whitespace=True).matches(SPACED)
    assert compile_schema(SCHEMA, whitespace=True).matches(VALID)
    assert not compile_schema(SCHEMA).matches(SPACED)


def test_integer_forms():
    nfa = compile_schema({"type": "integer"})
    assert all(nfa.matches(s) for s in ("0", "7", "-3", "120"))
    assert not any(nfa.matches(s) for s in ("", "-", "01", "1.5"))


def test_array_forms():
    nfa = compile_schema({"type": "array", "items": {"type": "integer"}})
    assert all(nfa.matches(s) for s in ("[]", "[1]", "[1,2,3]", "[1,2,3,4,5,6]"))
    assert not any(nfa.matches(s) for s in ("[,]", "[1,]", "[1 2]"))


def test_bounded_array_enforces_the_limit():
    nfa = compile_schema({"type": "array", "items": {"type": "boolean"}, "maxItems": 2})
    assert nfa.matches("[true,false]")
    assert not nfa.matches("[true,false,true]")


def test_unsupported_schema_raises():
    with pytest.raises(ValueError):
        compile_schema({"type": "number"})


def random_walk(mask, nfa, rng, vocab, limit=400):
    states, out = nfa.initial(), ""
    for _ in range(limit):
        options = mask.allowed(states)
        if nfa.accepts(states) and (not options or rng.random() < 0.3):
            return out
        token = rng.choice(sorted(options))
        out += vocab[token]
        states = options[token]
    raise AssertionError("no termination")


@pytest.mark.parametrize("seed", range(200))
def test_every_masked_walk_is_valid_json_in_the_schema(seed):
    rng = random.Random(seed)
    # A messy multi-character vocabulary, like a real tokenizer's.
    pieces = list(string.printable[:95]) + [
        '{"',
        '":"',
        '","',
        '":',
        "service",
        "severity",
        "minutes",
        "paged",
        "note",
        "tags",
        "checkout",
        "auth",
        "low",
        "high",
        "true",
        "false",
        "a",
        "b",
        '"a"',
        "ab",
        "hello",
        "12",
    ]
    vocab = dict(enumerate(pieces))
    nfa = compile_schema(SCHEMA)
    document = json.loads(random_walk(Mask(nfa, vocab), nfa, rng, vocab))
    assert list(document) == list(SCHEMA["properties"])
    assert document["service"] in ("checkout", "auth")
    assert isinstance(document["minutes"], int)
    assert len(document["note"]) <= 5


def test_mask_excludes_tokens_that_would_break_the_schema():
    nfa = compile_schema({"enum": ["yes"]})
    vocab = {0: '"y', 1: '"n', 2: '"yes"', 3: "x"}
    assert set(Mask(nfa, vocab).allowed(nfa.initial())) == {0, 2}
