"""Turn an automaton state into the set of vocabulary tokens that keep the output valid.

Walking every token from every state would be slow, so the vocabulary is a trie and walks share
prefixes. Results are memoised per automaton state, the same idea as the index in Outlines.
"""

from __future__ import annotations

from automaton import NFA


class Mask:
    def __init__(self, nfa: NFA, vocabulary: dict[int, str]):
        self.nfa = nfa
        self.trie: dict = {}
        for token_id, text in vocabulary.items():
            if not text:
                continue
            node = self.trie
            for char in text:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(token_id)
        self.memo: dict[frozenset[int], dict[int, frozenset[int]]] = {}

    def allowed(self, states: frozenset[int]) -> dict[int, frozenset[int]]:
        """Map each token that may come next to the automaton state it leads to."""
        if states not in self.memo:
            found: dict[int, frozenset[int]] = {}
            stack = [(self.trie, states)]
            while stack:
                node, here = stack.pop()
                for char, child in node.items():
                    if char is None:
                        found.update(dict.fromkeys(child, here))
                        continue
                    next_states = self.nfa.step(here, char)
                    if next_states:
                        stack.append((child, next_states))
            self.memo[states] = found
        return self.memo[states]
