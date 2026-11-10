"""Compile a small JSON Schema subset into an NFA over characters. Stdlib only.

Supported: object (all properties required, in schema order, no extra keys), string, enum, integer,
boolean, and array. Output is compact JSON with no whitespace.

A schema without recursion describes a regular language, so a finite automaton is enough. That is
why it can be precomputed and masked against the vocabulary at every step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

STRING_CHARS = frozenset(chr(c) for c in range(32, 127)) - {'"', "\\"}
DIGITS = frozenset("0123456789")
NONZERO = DIGITS - {"0"}


@dataclass
class NFA:
    edges: list[dict[str, set[int]]] = field(default_factory=list)
    epsilon: list[set[int]] = field(default_factory=list)
    start: int = 0
    accept: int = 0

    def state(self) -> int:
        self.edges.append({})
        self.epsilon.append(set())
        return len(self.edges) - 1

    def arrow(self, a: int, chars, b: int) -> None:
        for char in chars:
            self.edges[a].setdefault(char, set()).add(b)

    def closure(self, states) -> frozenset[int]:
        seen, stack = set(states), list(states)
        while stack:
            for nxt in self.epsilon[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen)

    def initial(self) -> frozenset[int]:
        return self.closure({self.start})

    def step(self, states: frozenset[int], char: str) -> frozenset[int]:
        return self.closure({t for s in states for t in self.edges[s].get(char, ())})

    def accepts(self, states: frozenset[int]) -> bool:
        return self.accept in states

    def matches(self, text: str) -> bool:
        states = self.initial()
        for char in text:
            states = self.step(states, char)
            if not states:
                return False
        return self.accepts(states)


def literal(nfa: NFA, start: int, text: str) -> int:
    for char in text:
        nxt = nfa.state()
        nfa.arrow(start, char, nxt)
        start = nxt
    return start


def _space(nfa: NFA, start: int, ws: bool) -> int:
    """Optional run of spaces, when the grammar allows whitespace."""
    if not ws:
        return start
    here = nfa.state()
    nfa.epsilon[start].add(here)
    nfa.arrow(here, " ", here)
    return here


def build(nfa: NFA, start: int, schema: dict, ws: bool = False) -> int:
    """Add the fragment for `schema` after state `start`. Returns the fragment's end state."""
    if "enum" in schema:
        end = nfa.state()
        for value in schema["enum"]:
            nfa.epsilon[literal(nfa, start, json.dumps(value))].add(end)
        return end
    kind = schema.get("type")
    if kind == "boolean":
        return build(nfa, start, {"enum": [True, False]})
    if kind == "integer":
        end, first, zero = nfa.state(), nfa.state(), nfa.state()
        nfa.epsilon[start].add(first)
        minus = nfa.state()
        nfa.arrow(start, "-", minus)
        nfa.epsilon[minus].add(first)
        nfa.arrow(first, "0", zero)
        nfa.epsilon[zero].add(end)
        rest = nfa.state()
        nfa.arrow(first, NONZERO, rest)
        nfa.arrow(rest, DIGITS, rest)
        nfa.epsilon[rest].add(end)
        return end
    if kind == "string":
        opened = literal(nfa, start, '"')
        limit = schema.get("maxLength")
        if limit is None:
            nfa.arrow(opened, STRING_CHARS, opened)
            return literal(nfa, opened, '"')
        end, here = nfa.state(), opened
        for _ in range(limit):
            nfa.arrow(here, '"', end)
            nxt = nfa.state()
            nfa.arrow(here, STRING_CHARS, nxt)
            here = nxt
        nfa.arrow(here, '"', end)
        return end
    if kind == "array":
        return _array(nfa, start, schema, ws)
    if kind == "object":
        here = _space(nfa, literal(nfa, start, "{"), ws)
        for i, (key, value) in enumerate(schema["properties"].items()):
            if i:
                here = _space(nfa, literal(nfa, here, ","), ws)
            here = _space(nfa, literal(nfa, here, json.dumps(key) + ":"), ws)
            here = _space(nfa, build(nfa, here, value, ws), ws)
        return literal(nfa, here, "}")
    raise ValueError(f"unsupported schema: {schema}")


def _array(nfa: NFA, start: int, schema: dict, ws: bool = False) -> int:
    opened = _space(nfa, literal(nfa, start, "["), ws)
    end = nfa.state()
    limit = schema.get("maxItems")
    if limit is None:
        nfa.arrow(opened, "]", end)
        item_end = _space(nfa, build(nfa, opened, schema["items"], ws), ws)
        nfa.arrow(item_end, "]", end)
        comma = nfa.state()
        nfa.arrow(item_end, ",", comma)
        after_comma = _space(nfa, comma, ws)
        # Two copies of the item at any length: the second one loops back on itself.
        repeat_end = _space(nfa, build(nfa, after_comma, schema["items"], ws), ws)
        nfa.arrow(repeat_end, ",", comma)
        nfa.arrow(repeat_end, "]", end)
        return end
    # Bounded repetition has to copy the item once per position.
    here, first = opened, True
    for _ in range(limit):
        nfa.arrow(here, "]", end)
        if not first:
            separator = nfa.state()
            nfa.arrow(here, ",", separator)
            here = _space(nfa, separator, ws)
        here = _space(nfa, build(nfa, here, schema["items"], ws), ws)
        first = False
    nfa.arrow(here, "]", end)
    return end


def compile_schema(schema: dict, whitespace: bool = False) -> NFA:
    """`whitespace` allows spaces where JSON does, after brackets, colons and commas."""
    nfa = NFA()
    nfa.start = nfa.state()
    nfa.accept = build(nfa, nfa.start, schema, whitespace)
    return nfa
