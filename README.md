# constrained-decoding-from-scratch

JSON-Schema-constrained decoding from scratch: compile a schema to a finite automaton, mask a real tokenizer's vocabulary at every step, and measure what guaranteed-valid output does and doesn't buy you. Standard library only, apart from the experiment.

Companion to [Structured Outputs Guarantee the Shape, Not the Answer](https://cloudandsre.com/blog/constrained-decoding-from-scratch/).

## How it works

1. `automaton.py` compiles a schema subset (objects with ordered required properties, strings, enums, integers, booleans, arrays) into an NFA over characters. A schema without recursion is a regular language, which is why this is possible at all.
2. `mask.py` walks the tokenizer's vocabulary as a trie from the current automaton state, so tokens sharing a prefix share the work, and memoises the allowed set per state. This is the idea behind [Willard and Louf's](https://arxiv.org/abs/2307.09702) Outlines.
3. `experiment.py` decodes with the mask applied to a real model's logits, and compares against prompting alone.

## Results

| Mode | Valid | All fields right | service | severity | minutes | Mask cost per token |
|---|---:|---:|---:|---:|---:|---:|
| prompt only | 3% | 3.0% | 3.0% | 20.5% | 86% | 0.000 ms |
| constrained, compact grammar | 100% | 40.0% | 40.0% | 100.0% | 99% | 0.027 ms |
| constrained, whitespace allowed | 100% | 40.0% | 40.0% | 100.0% | 99% | 0.030 ms |
| constrained, cut off at 8 tokens | 0% | 0.0% | 0.0% | 0.0% | 0% | 0.026 ms |

Mask setup for a 48,838-token vocabulary: 172 ms. Bounded repetition copies automaton states, which is why providers reject `maxLength` and `maxItems`:

| Limit | Unbounded array | Array with maxItems | Nested, both bounded |
|---:|---:|---:|---:|
| 5 | 58 | 142 | 712 |
| 10 | 68 | 332 | 3,322 |
| 20 | 88 | 862 | 17,242 |
| 40 | 128 | 2,522 | 100,882 |

Full output in [`results/experiment.json`](results/experiment.json).

## Run it

```bash
pip install -r requirements-ci.txt
pytest -q                  # includes 200 random masked walks, each must parse and fit the schema

pip install torch transformers
python experiment.py       # downloads SmolLM2-135M-Instruct, about 270 MB
```

## Limits

The schema subset is small and the automaton is character-level, so it's slower than production grammar engines. Tokens that decode to partial UTF-8 are excluded from the vocabulary, which rules out non-ASCII strings. It's a teaching implementation, and the measurements are on a 135M-parameter model.

## License

Apache-2.0
