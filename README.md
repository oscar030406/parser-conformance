# parser-conformance

parser-conformance finds bugs in the tool-call and reasoning parsers of [vLLM](https://github.com/vllm-project/vllm). It records every model output that vLLM's own parser tests pass to a parser. It then changes each output in ways that real traffic does, and checks properties that every parser should keep.

vLLM has about 60 tool-call parsers and 40 reasoning parsers, one or more per model family. Each parser has its own hand-written tests, and on vLLM `main` at `191e5d5` only 8 of the 38 tool-parser test files use the shared test suite in `tests/tool_parsers/common_tests.py`. The same kind of bug therefore shows up in parser after parser. This plugin applies one set of checks to every parser that has tests, with no per-parser configuration.

## How it works

1. A pytest plugin wraps `extract_tool_calls` and `extract_reasoning` on every parser class. While vLLM's tests run, it records each parser instance, the model output it received, and the request.
2. After the tests finish, it checks each recorded output, plus these variants of it:
   - each token-boundary prefix, which is what a `max_tokens` cut produces
   - the output with a plain sentence put in front of it or after it
   - the output fed token by token through the streaming path.

Tool-call parsers:

| Check | Property |
|---|---|
| `I1_flag` | `tools_called` is true exactly when `tool_calls` is non-empty. |
| `I2_text` | When the parser finds no call, `content` keeps the model output. |
| `I3_calls`, `I3_content` | Streaming gives the same calls and content as non-streaming. |
| `I3_calls_chunked`, `I3_content_chunked` | The same holds when each delta carries 2 to 8 tokens, as with speculative decoding. |
| `prefix_lost`, `prefix_calls` | A sentence in front of the output reaches `content` and leaves the calls unchanged. |
| `prefix_anchored` | The parser finds no call once a sentence comes before it. |
| `S_state` | One parser instance parses the output and all its variants in turn, and gives the same result as a fresh parser for each. |

Reasoning parsers:

| Check | Property |
|---|---|
| `R1_prefix`, `R1_suffix` | The parser keeps a sentence in front of or after the output, in reasoning or in content. |
| `R2_stream` | Streaming gives the same reasoning and content as non-streaming. |
| `R2_trunc` | The same holds at every token-boundary prefix. |

A `_trunc` suffix on a tool-call check means it failed on a prefix of the recorded output. A `_markers` variant, such as `I2_markers` or `R2_stream_markers`, means the two results differ only in special tokens or whitespace. Every check except `S_state` builds a fresh parser for each input, as vLLM does for each request.

## Requirements

- A vLLM source checkout with its development environment. The plugin imports vLLM's test helpers in `tests/tool_parsers/utils.py`.
- Python 3.10 or later. CPU only.
- The tokenizers that vLLM's tests download from the Hugging Face Hub. Set `HF_ENDPOINT` to use a mirror.

## Install

Install the plugin into the environment of the vLLM checkout:

```bash
git clone https://github.com/oscar030406/parser-conformance
pip install -e ./parser-conformance
pip install pytest-xdist   # optional, for -n
```

The plugin has no dependencies beyond vLLM's test environment.

## Usage

Run from the root of the vLLM checkout, so that `python -m pytest` can import vLLM's `tests` package:

```bash
PC_REPORT=/tmp/pc.jsonl python -m pytest tests/tool_parsers tests/reasoning -n 8 \
    -p parser_conformance.plugin -q
python -m parser_conformance.summarize /tmp/pc.jsonl --examples
```

vLLM's tests run as usual. With `-n`, each worker checks the outputs it recorded and hands the rows to the main process, which writes one report. An error inside the checks fails the run and adds a `harness_error` row. The summary prints one row per parser: the number of recorded outputs (`seed`), then a count per check. It then prints the shortest failing input for each pair. For example, on `191e5d5`:

```text
parser                 seed  I1_flag_trunc  ...  policy_drops_all  policy_drops_call  ...
DeepSeekV31ToolParser  3     44             ...  26                18                 ...

## DeepSeekV31ToolParser I1_flag_trunc
{"input": "<｜tool▁calls▁begin｜>", "result": {"tools_called": true, "calls": [], "content": null}}
```

`PC_MAX_SEEDS` (default 40) limits the recorded outputs per parser. With `-n`, the report keeps the outputs with the first test ids. `PC_MAX_CUTS` (default 48) limits the prefixes checked per output.

## Results on vLLM main

Run on `191e5d5` (2026-11-13) with version 0.2.0. The tool-call tests take 2 to 3 minutes with `-n 8`, and the reasoning tests take 4 minutes without `-n`. Each row is a bug that the checks found without any parser-specific setup.

| Parser | Bug | Check | Upstream |
|---|---|---|---|
| `deepseek_v31` | A tool-calls section with no parsable call sets `tools_called` and drops the whole output from non-streaming `content` | `I1_flag`, `policy_drops_all` | [vllm#67491](https://github.com/vllm-project/vllm/pull/67491) (fix) |
| `hy_v3` | Same as above | `I1_flag`, `policy_drops_all` | [vllm#67496](https://github.com/vllm-project/vllm/pull/67496) (fix) |
| `deepseek_v3` | Same as above | `I1_flag`, `policy_drops_all` | [vllm#48367](https://github.com/vllm-project/vllm/pull/48367) |
| `ernie45` | Same as above | `I1_flag`, `policy_drops_all` | [vllm#65346](https://github.com/vllm-project/vllm/pull/65346) |
| `internlm2` | Malformed JSON raises `JSONDecodeError` | `raises_trunc` | [vllm#50078](https://github.com/vllm-project/vllm/pull/50078) |
| `deepseek_r1` (`BaseThinkingReasoningParser`) | Drops the text before `<think>` | `R1_prefix` | [vllm#50901](https://github.com/vllm-project/vllm/issues/50901) |
| `hunyuan_a13b` | Drops the text before `<think>` when `</answer>` is missing | `R1_prefix` | [vllm#67509](https://github.com/vllm-project/vllm/pull/67509) (fix) |
| `hunyuan_a13b` | Drops the text after `</answer>` | `R1_suffix` | not reported yet |

The two fixes linked above remove their findings. On the fix branches, `I1_flag` drops to 0 for `deepseek_v31` and `hy_v3`, and every cut inside a call moves to `policy_keeps_text`.

## Truncated tool calls

When `max_tokens` cuts an output in the middle of a tool call, no complete call remains. vLLM's parsers then disagree on the response. The plugin records each such cut as a `policy_*` row instead of a failure. A cut counts as inside a call when it contains a marker of the parser and the full output has a call.

Each row has two results. The check name gives what `extract_tool_calls` returns: `keeps_text`, `drops_call` (only the text before the call), `drops_all` (no content) or `ghost_call` (a call that the full output does not have). The `served_*` fields give what the server returns: vLLM's `Parser.parse` with only this tool parser. The two can differ. When `tools_called` is false, `Parser.parse` returns the raw output as content for parsers that are not built on vLLM's parser engine.

The suffix of the check name gives the cut position. `@opener` cuts stop in the markup before the function name. `@call` cuts reach the name.

What the server returns, on `0fda86a` (2026-11-20), as cut positions:

| `--tool-call-parser` | cut before the name | cut at or after the name |
|---|---|---|
| `xlam`, `lfm2`, `longcat`, `jamba`, `olmo3`, `phi4_mini_json`, `granite-20b-fc`, `granite4`, `hermes`, `step3`, `internlm`, `llama4_pythonic`, `gigachat3`, `deepseek_v3`, `deepseek_v31`, `ernie45` | keeps the cut output | keeps the cut output |
| `kimi_k2` | drops the cut call (29) | drops the cut call (279) |
| `granite` | drops the cut call (43) | drops the cut call (166) |
| `granite_toolcall_parser` | drops the cut call (31) | drops the cut call (160) |
| `glm45`, `glm47` | drops the cut call (9) | returns a call in 38 of 65 |
| `qwen3_coder`, `qwen3_xml` | drops the cut call (87) | returns a call in 147 of 673, 6 of them with a cut name |
| `hy_v3` | drops the cut call (16) | returns a call in 73 of 90 |
| `step3p5` | drops the cut call (36) | returns a call in 309 of 395, 20 of them with a cut name |

"Drops the cut call" keeps the text before the call. A returned call has the arguments that the cut output holds, for example `{}` for `<function=get`, and the server reports `finish_reason` `length`. The DeepSeek v32, v4 and v41 tests use a stand-in tokenizer, so their cuts are not in the table.

`pythonic` has no marker, so its cuts are not in the table. `I2_text` checks them and finds no lost text. Which behavior vLLM intends is the open question in [vllm#67513](https://github.com/vllm-project/vllm/issues/67513).

## Comparing with vLLM's Rust frontend

vLLM is porting its parsers to a Rust frontend in `rust/src/parser`. [`rust/parser_diff.rs`](rust/parser_diff.rs) is a small program for vLLM's `rust/src/chat` crate. It reads the recorded outputs and runs the Rust parser of the same name on each one. It feeds the text through the tokenizer and the incremental decoder, as the frontend does. `parser_conformance.rust_diff` then compares the Rust results with the Python ones.

```bash
cp rust/parser_diff.rs /path/to/vllm/rust/src/chat/examples/
(cd /path/to/vllm/rust && cargo build -p vllm-chat --example parser_diff)
# from the vLLM checkout, record the outputs with the Python results
PC_REPORT=/tmp/pc.jsonl PC_SEEDS_OUT=/tmp/seeds.jsonl python -m pytest tests/tool_parsers -n 8 \
    -p parser_conformance.plugin -q
python -m parser_conformance.rust_diff rust/target/debug/examples/parser_diff /tmp/seeds.jsonl
```

On `191e5d5`, 115 recorded outputs have a parser in both implementations and a `tokenizer.json` in the local cache. 74 of them parse the same. The other 41 fall into these groups:

- 26 are errors on the Rust side.
  - In 7 of them, a cut leaves an unfinished call. The Rust parser's `finish()` reports it, and the frontend drops the unfinished markup. The Python parsers of the same names return the cut output as content. See [vllm#67513](https://github.com/vllm-project/vllm/issues/67513).
  - In 7, Rust's `glm45` expects a newline after the function name. In Python, `glm45` is another name for the GLM-4.7 parser, which also accepts output without it.
  - In the other 12, the Rust parsers reject malformed outputs from the Qwen3, Phi-4 and DeepSeek V3 tests.
- 12 are Qwen3 XML outputs with a missing `</parameter>`, from `qwen3_coder` and `qwen3_xml`. Python ends the value at the next `<parameter=`. Rust keeps that tag in the value, so `city` becomes `"Dallas\n<parameter=state>\nTX"` and `state` is lost.
- 2 are `phi4_mini_json` outputs that Rust parses. For one, Python sets `tools_called` with no call. For the other, Python drops the text around the call.
- 1 is a DeepSeek V3 output with text after `<｜tool▁calls▁end｜>`. The Rust parser ignores that text, and its tests assert this. Python keeps it.

The export builds a fresh Python parser for each output, and it sends the tool schemas that engine parsers take in `__init__`. Without these, Python and Rust disagree on outputs that both parse the same way.

`S_state` reports one parser on `191e5d5`: after the cut outputs, `granite4` leaves `in_tc` set, and the same instance then returns a complete call as content. vLLM builds a new parser for each choice, so served requests do not reach this.

## Streaming cost

Set `PC_PERF=1` to time streaming as an output grows. For each tool parser, the plugin takes its longest recorded output. It puts 256 or 1024 tokens of plain text into the first string value of the arguments. When the output has no such value, the text goes in front of it. A fresh parser then streams each version token by token. The time is the fastest of 3 runs and leaves out building the tokenizer vocabulary. A time ratio near 4 means constant work per token. A ratio near 16 means every token rescans the whole text.

```bash
PC_PERF=1 PC_PERF_OUT=/tmp/perf.jsonl PC_MAX_CUTS=1 python -m pytest tests/tool_parsers -n 8 \
    -p parser_conformance.plugin -q
```

On `191e5d5`, 19 parsers give ratios from 2.7 to 4.5. Four give more:

| `--tool-call-parser` | 256 tokens | 1024 tokens | Ratio |
|---|---|---|---|
| `granite` | 98 ms | 925 ms | 9.5 |
| `jamba` | 36 ms | 305 ms | 8.6 |
| `granite-20b-fc` | 42 ms | 328 ms | 7.9 |
| `xlam` | 23 ms | 161 ms | 7.0 |

The streaming path of `xlam` runs regular expressions and `json.loads` over the whole text for every token. `ernie45` fails the streaming harness's own assertion, so it has no time.

## Checking the tool

The bugs above serve as controls. On a commit that has a bug, its row must show findings. On a commit with the fix, the row must show none. Check that the `seed` column is above 0 first, because a parser with no recorded output also shows no findings:

```bash
git switch fix-deepseek-v31-unparsed-tool-calls   # branch of vllm#67491
PC_REPORT=/tmp/neg.jsonl python -m pytest tests/tool_parsers/test_deepseekv31_tool_parser.py \
    -p parser_conformance.plugin -q
python -m parser_conformance.summarize /tmp/neg.jsonl   # seed above 0, no I1_flag column
```

## Limitations

- Coverage follows vLLM's tests. A parser with no test is not checked. Some tests build the parser with a stand-in tokenizer, such as a `MagicMock` with an empty vocabulary. Streaming then cannot see the marker token ids, so the plugin runs only the full-output and prefix checks and reports `skip_mock_tokenizer`. On `191e5d5`, 16 tool-parser test files do this.
- A finding is a candidate, not a verdict. vLLM documents some differences as design:
  - `minicpm5_xml` drops content on purpose when it parses a call, and its tests assert this.
  - The `gemma4` reasoning tests expect different results per mode for output without a start marker.
  - Incremental streaming cannot take back a call that it started. A call that never closes therefore appears only in streaming. The plugin reports this as `I3_lenient` when the extra calls come last and no closing marker, such as `</tool_call>`, follows their names. For formats without closing markers, it checks that their arguments are not complete JSON.
- `I2_text` ignores the markers that it can find. These are special and added tokens with tag characters such as `<`, `|` or `[`, and `*_token` attributes of the parser and its classes. They are also the terminals of parsers built on vLLM's parser engine, and tag-like strings such as `<answer>` in the parser's source. It does not know marker phrases such as Granite's `Here is my thought process:`, which is why the reasoning checks only compare a parser with itself.
- The plugin compares the streaming state without an end-of-stream flush. Most `R2_trunc` findings are a parser holding back the start of a possible marker at the cut, for example `<th` or Granite's `Here`, which non-streaming returns as content.

## License

Apache-2.0, the same as vLLM. See [LICENSE](LICENSE).
