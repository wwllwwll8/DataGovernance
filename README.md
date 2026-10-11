# df_distilabel

A small Python package that stacks **Distilabel-style LLM task recipes** on a **DataFlow-style operator pipeline**.

[DataFlow](https://github.com/OpenDCAI/DataFlow) treats data preparation as operators over a dataframe. Each operator reads a step file, writes the next step, and names the columns it reads and writes with `input_*` / `output_*` arguments. `Pipeline.compile()` turns that into a static graph and checks the columns before any model call. Column edges are `KeyNode`s (producer → consumer).

[Distilabel](https://github.com/argilla-io/distilabel) treats data preparation as LLM tasks: text generation, preference judges such as UltraFeedback, and a dataset artifact (a Distiset) you can push to Argilla or the Hugging Face Hub.

This package keeps both ideas and vendors neither codebase. v0 is a single-process prototype: JSONL step files, a compile-time column check, a text lineage dump, two task recipes, and objective verify gates. Tests run offline with `MockLLM`.

```
entry JSONL (step 0)
        │
        ▼
LoadJSONL ──► TextGeneration ──► UltraFeedback-like judge
                                        │
                                        ▼
                                 HashDeduplicate ──► MathVerify ──► JSONL export
        ▲
        Pipeline.compile() checks columns before any LLM call
```

## Layout

| Path | Role |
| --- | --- |
| `df_distilabel/core/operator.py` | `OperatorABC.run(storage, ...)`, registry |
| `df_distilabel/core/storage.py` | `FileStorage` JSONL step chain |
| `df_distilabel/core/pipeline.py` | `compile` / `forward` / `run`, `AutoOP`, `KeyNode`, `draw` |
| `df_distilabel/llm.py` | `LLMClient` protocol and `MockLLM` |
| `df_distilabel/ops/load.py` | `LoadJSONL` |
| `df_distilabel/ops/distilabel_bridge.py` | Text generation and UltraFeedback-like scoring |
| `df_distilabel/ops/verify.py` | Hash dedupe, math/exact check, SELECT-only SQL stub |
| `examples/pref_then_verify.py` | Offline demo |
| `tests/` | `pytest`, no network and no API keys |

## Install and run

From the repo root, with Python 3.11+:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 examples/pref_then_verify.py
```

`pandas` is the dataframe type, as in DataFlow. `pytest` is the test runner. `math-verify` and `distilabel` are optional extras; the default path does not import them.

## Operator contract

Subclass `Pipeline`, put one `FileStorage` and the operators on `self`, and implement `forward` as a straight line of `run` calls. Operators have to be attributes of the pipeline so compile can see them.

```python
class PrefThenVerify(Pipeline):
    def __init__(self, cache_path, llm):
        super().__init__()
        self.storage = FileStorage(cache_path=cache_path)
        self.load = LoadJSONL("examples/data/seeds.jsonl")
        self.gen = TextGenerationOp(llm)

    def forward(self):
        self.load.run(self.storage.step())
        self.gen.run(
            self.storage.step(),
            input_key="instruction",
            output_key="generation",
        )
```

`storage.step()` follows the DataFlow counter. It starts at `-1`, advances by one, and hands that operator a view:

- Step 0 reads `first_entry_file_name` (the entry JSONL). An empty entry is an empty frame, which is what you want when `LoadJSONL` is the source.
- `write()` saves step `operator_step + 1` as `step1.jsonl`, `step2.jsonl`, ... in the cache directory.
- The next operator's view reads that file.

Column arguments start with `input_` or `output_`. A string is one column name. A list of strings (`input_keys=["instruction", "generation"]`) is several. Other arguments, such as `drop_failures=True`, are runtime options.

## Compile, before any LLM call

`pipeline.compile()` (also invoked by `run()` and `draw()`):

1. Wraps each operator attribute in `AutoOP`.
2. Calls your `forward()`. `AutoOP.run` records the storage view and the column arguments. It does not call `Operator.run` and does not call the LLM.
3. Checks every input column against the entry file or the outputs of earlier operators.
4. Builds `KeyNode` links from each producer to the operators that read the column.
5. Replaces `forward` with the real execution of the recorded calls.

A typo raises `KeyError` naming the operator, the parameter, and the columns that do exist:

```text
Key Matching Error in following Operators during pipeline.compile():
- Input key 'instructon' in `gen` (class <TextGenerationOp>) ...
  Check parameter 'input_key' in `gen.run()`.
Compile stopped before any operator.run() or LLM call.
```

`LoadJSONL` is a source. Compile reads its column names from the file (local I/O, not an LLM) and it must be the first step.

## Lineage

`pipeline.draw()` returns a text dump. `KeyNode.readers` is the producer → consumer edge. A column is **dead** when no later operator reads it. Terminal products (judge scores, a verify flag) show up there too: they are exported on the row and unused by the rest of the graph. An unread seed column or a scratch column written in the middle is the same list, which makes typos and forgotten fields visible.

The demo's dump looks like this:

```text
pipeline: load:LoadJSONL -> gen:TextGenerationOp -> judge:UltraFeedbackOp -> dedup:HashDeduplicateFilter -> math:MathVerifyFilter
column lineage
instruction: load:LoadJSONL -> gen:TextGenerationOp, judge:UltraFeedbackOp, dedup:HashDeduplicateFilter
answer: load:LoadJSONL -> math:MathVerifyFilter
generation: gen:TextGenerationOp -> judge:UltraFeedbackOp, dedup:HashDeduplicateFilter, math:MathVerifyFilter
ratings: judge:UltraFeedbackOp -> (none)
rationales: judge:UltraFeedbackOp -> (none)
overall: judge:UltraFeedbackOp -> (none)
math_correct: math:MathVerifyFilter -> (none)
dead columns: ratings, rationales, overall, math_correct
```

DataFlow can render this graph with networkx. v0 stops at the text dump.

## Task recipes and the LLM client

`LLMClient.generate(chats)` takes a batch of chats (`{"role", "content"}` messages) and returns one string per chat, in order. `MockLLM` replays a script or a Python function and records every call. No network.

`TextGenerationOp` reads `input_key` (default `instruction`) and writes `output_key` (default `generation`).

`UltraFeedbackOp` reads an instruction and one generation. It asks the client for JSON scores on the UltraFeedback aspects helpfulness, honesty, instruction-following, and truthfulness, and writes:

- `ratings` — aspect → score
- `rationales` — aspect → text
- `overall` — the judge's overall, or the mean of the aspect scores

The real Distilabel task is different in one important way: `distilabel.steps.tasks.UltraFeedback` scores a **list** of generations in one prompt and returns parallel `ratings` / `rationales` lists plus `model_name`, for an aspect such as `overall-rating`. The parser accepts that list-of-floats shape (mean → `overall`) so a later adapter has somewhere to land. The operator itself still sends one generation per row.

### Optional Distilabel object

Pass any object with `inputs`, `outputs`, `format_input(record) -> chat`, and `format_output(text, record) -> dict`. `TextGenerationOp(llm, task=that_object)` uses those columns and does not import `distilabel`. `distilabel_available()` reports whether the package is installed. Installing it does not switch the built-in recipes over; that wiring is a TODO.

```python
# TODO(distilabel): construct distilabel.steps.tasks.UltraFeedback(llm=..., aspect=...)
# and map its ratings / rationales / model_name columns onto this operator.
```

## Verify gates

These operators do not call an LLM.

- `HashDeduplicateFilter` hashes `input_key` or `input_keys` (whitespace-normalized strings) and keeps the first row of each group.
- `MathVerifyFilter` compares `input_key` to `input_gold_key`. It uses `math_verify` when that package imports, then normalized exact text, then the last number on each side (including `\boxed{...}`). Failures are dropped unless `drop_failures=False`. The boolean lands in `output_key` (default `math_correct`).
- `SQLExecFilter` is a stub. Trusted `setup_sql` runs on a fresh in-memory sqlite database per row. SQL from the row must be a single `SELECT` or `WITH ... SELECT`. Writes and multi-statements are failures. At most 50 result rows are stored.

## Export

`export_jsonl(path)` writes the last step and a sidecar `*.manifest.json` (`kind: distiset-like`) with columns, operators, and the lineage text.

```python
# TODO(argilla): Pipeline.export_argilla should call
# distilabel.steps.PreferenceToArgilla (instruction, generations,
# optional ratings and rationales). Distiset.push_to_hub publishes
# the same artifact to the Hugging Face Hub.
```

`export_argilla()` raises `NotImplementedError` until that adapter exists.

## Demo

`examples/pref_then_verify.py` loads five math seeds (one duplicate `2 + 2`, one wrong `9 + 1` draft). `MockLLM` answers the others correctly. The judge scores all five rows, including the wrong one. Hash dedupe drops the duplicate. `MathVerifyFilter` drops `9 + 1`. Three rows remain, written to `examples/output/distiset.jsonl`.

That order matches the sketch above: the judge runs before the objective gates. A later scheduler can move hash and math earlier; v0 runs steps in the order `forward` records them.

## Non-goals

v0 does not vendor OpenDCAI/DataFlow or Distilabel, does not run Ray, and does not ship a live OpenAI (or other) client. Bring your own `LLMClient` when you want a real model. Tests and the demo stay on `MockLLM`.
