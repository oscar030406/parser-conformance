# Changelog

## [Unreleased]

## [0.3.0] - 2026-11-20

### Added

- Each `policy_*` row has the cut position as a suffix: `@opener` for a cut before the function name, `@call` for a cut at or after it.
- A cut that returns a call the full output does not have is a `policy_ghost_call` row. Its `cut_name` field tells whether the name is cut.
- The `served_content`, `served_calls` and `served_kind` fields give what vLLM's `Parser.parse` returns for the cut, the path that serves non-streaming chat. For parsers that are not built on vLLM's parser engine, it returns the raw output when `tools_called` is false, so it can differ from `extract_tool_calls`.

## [0.2.0] - 2026-11-16

### Fixed

- Every parse uses a fresh parser built with the arguments of the test. Before, one instance parsed the output and all its variants. A parser that keeps state between calls then produced findings that served requests never hit.
- `S_state` now parses the output and every variant with one instance in turn, and compares each result with a fresh parser. It reports the first input that differs.
- Streaming no longer splits a character of several bytes over two deltas. Like vLLM's detokenizer, a delta waits until the character is complete.
- `I3_calls` no longer treats `""` and `"{}"` arguments as equal.
- The plugin reports `I3_lenient` only when streaming returns the non-streaming calls plus calls at the end that never close. A call never closes when no closing marker, such as `</tool_call>`, follows its name. For formats without closing markers, a call never closes when its arguments are not complete JSON. The plugin reports other differences as `I3_calls`.
- Before, the plugin reported a parser that missed a complete call in non-streaming as lenient. Step3 is an example: only its non-streaming path needs `function<｜tool_sep｜>`.
- A cut inside a call that leaves no complete call gets the `I1_flag` check and a `policy_*` row, but no `I2_text` check. A cut before the call gets every check. Before, the plugin counted cuts before the call as `policy_keeps_text`. That raised the count for parsers that drop the cut output.
- The plugin also takes markers from the class attributes of a parser and from tag-like strings in its source. For parsers built on vLLM's parser engine, it takes the terminals of the engine. Before, it reported `granite` and `llama4_pythonic` cuts that kept the text without the marker as `I2_text`.
- A seed is one output with one tokenizer, request and set of constructor tools. The key leaves out vLLM's random `request_id`. `PC_MAX_SEEDS` holds for the whole run, also with `-n`.
- Only markers that are in the output but not in its content mark a call. Added tokens count as markers only when they have tag characters, so ERNIE's digits do not.
- With pytest-xdist, the workers send their rows to the controller. The controller writes one report, one seed export and one streaming-cost file. Before, each worker wrote its own file, and the files of some workers were empty.
- An error inside the checks fails the run and writes a `harness_error` row. Before, a worker raised the error from `pytest_sessionfinish`, where pytest-xdist does not report it.
- The seed export writes a row for a parser that raises, with the error in `python.error`. It skips registered parsers whose optional dependency is not installed. It writes `tokenizer` only when it is a name.
- The streaming-cost timer starts after the parser reads the tokenizer vocabulary, and it takes the fastest of 3 runs. The padding goes into the first argument value.
- `rust_diff` treats only "is not registered" as a parser that Rust lacks. It skips seeds where the Python parser raised, and it leaves the stderr of the Rust program on the terminal.
- `parser_diff.rs` reports a tokenizer that does not load as an error row, and it prints the full chain of each error.

### Added

- `I3_markers`, `R2_stream_markers` and `R2_trunc_markers`: the two modes differ only in special tokens or whitespace.
- `prefix_anchored`: the parser finds no call once text comes before it.
- `I3_calls_chunked` and `I3_content_chunked`: streaming with 2 to 8 tokens per delta, as speculative decoding produces.
- A `seed` row in the report for each recorded output. `summarize` shows the count in its first column.
- Tests on a fake vLLM checkout in `tests/fake_vllm`, `check.sh`, and a CI workflow.

## [0.1.0] - 2026-11-13

- First release: the tool-call and reasoning checks, the seed export, the Rust comparison and the streaming-cost timing.
