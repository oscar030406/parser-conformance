# parser-conformance

parser-conformance finds bugs in the tool-call and reasoning parsers of [vLLM](https://github.com/vllm-project/vllm). It records every model output that vLLM's own parser tests pass to a parser, changes each output in ways that real traffic does, and checks properties that every parser should keep.

vLLM has about 60 tool-call parsers and 40 reasoning parsers, one or more per model family. Each parser has its own hand-written tests, and on vLLM `main` at `191e5d5` only 8 of the 38 tool-parser test files use the shared test suite in `tests/tool_parsers/common_tests.py`. The same kind of bug therefore shows up in parser after parser. This plugin applies one set of checks to every parser that has tests, with no per-parser configuration.

## How it works

1. A pytest plugin wraps `extract_tool_calls` and `extract_reasoning` on every parser class. While vLLM's tests run, it records each parser instance, the model output it received, and the request.
2. After the tests finish, it checks each recorded output, plus these variants of it:
   - each token-boundary prefix, which is what a `max_tokens` cut produces;
   - the output with a plain sentence put in front of it or after it;
   - the output fed token by token through the streaming path.

Tool-call parsers:

| Check | Property |
|---|---|
| `I1_flag` | `tools_called` is true exactly when `tool_calls` is non-empty. |
| `I2_text` | When no call is parsed, `content` keeps the model output. |
| `I3_calls`, `I3_content` | Streaming gives the same calls and content as non-streaming. |
| `prefix_lost`, `prefix_calls` | A sentence in front of the output reaches `content` and leaves the calls unchanged. |
| `S_state` | After all the variants, the same parser instance parses the original output as it did first. |

Reasoning parsers:

| Check | Property |
|---|---|
| `R1_prefix`, `R1_suffix` | A sentence in front of or after the output is kept, in reasoning or in content. |
| `R2_stream` | Streaming gives the same reasoning and content as non-streaming. |
| `R2_trunc` | The same holds at every token-boundary prefix. |

A `_trunc` suffix on a tool-call check means it failed on a prefix of the recorded output.

## Requirements

- A vLLM source checkout with its development environment. The plugin imports vLLM's test helpers in `tests/tool_parsers/utils.py`.
- Python 3.10 or later. CPU only.
- The tokenizers that vLLM's tests download from the Hugging Face Hub. Set `HF_ENDPOINT` to use a mirror.

## Install

```bash
git clone https://github.com/oscar030406/parser-conformance
pip install pytest-xdist   # optional, for -n
```

The plugin has no dependencies beyond vLLM's test environment. Put the clone on `PYTHONPATH`.

## Usage

Run from the root of the vLLM checkout:

```bash
PYTHONPATH=$PWD:/path/to/parser-conformance PC_REPORT=/tmp/pc.jsonl \
    python -m pytest tests/tool_parsers tests/reasoning -n 8 -p parser_conformance.plugin -q
python -m parser_conformance.summarize /tmp/pc.jsonl* --examples
```

vLLM's tests run as usual. With `-n`, each worker writes its own report file, and `summarize` merges them. The summary prints one row per parser with a count per check, then the shortest failing input for each pair. For example, on `191e5d5`:

```text
parser                 I1_flag_trunc  I2_markers_trunc  I2_text_trunc  ...
DeepSeekV31ToolParser  44             6                 38             ...

## DeepSeekV31ToolParser I2_text_trunc
{"input": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>foo", "result": {"tools_called": true, "calls": [], "content": null}, ...}
```

`PC_MAX_SEEDS` (default 40) limits the recorded outputs per parser. `PC_MAX_CUTS` (default 48) limits the prefixes checked per output.

## Results on vLLM main

Run on `191e5d5` (2026-11-13). The tool-call tests take 3 minutes with `-n 8`, and the reasoning tests take 7 minutes without `-n`. Each row is a bug that the checks found without any parser-specific setup.

| Parser | Bug | Check | Upstream |
|---|---|---|---|
| `deepseek_v31` | A tool-calls section with no parsable call drops the whole output from non-streaming `content` | `I1_flag`, `I2_text` | [vllm#67491](https://github.com/vllm-project/vllm/pull/67491) (fix) |
| `hy_v3` | Same as above | `I1_flag`, `I2_text` | [vllm#67496](https://github.com/vllm-project/vllm/pull/67496) (fix) |
| `deepseek_v3` | Same as above | `I1_flag`, `I2_text` | [vllm#48367](https://github.com/vllm-project/vllm/pull/48367) |
| `ernie45` | Same as above | `I1_flag`, `I2_text` | [vllm#65346](https://github.com/vllm-project/vllm/pull/65346) |
| `internlm2` | Malformed JSON raises `JSONDecodeError` | `raises_trunc` | [vllm#50078](https://github.com/vllm-project/vllm/pull/50078) |
| `deepseek_r1` (`BaseThinkingReasoningParser`) | Text before `<think>` is dropped | `R1_prefix` | [vllm#50901](https://github.com/vllm-project/vllm/issues/50901) |
| `hunyuan_a13b` | Text before `<think>` is dropped when `</answer>` is missing | `R1_prefix` | [vllm#67509](https://github.com/vllm-project/vllm/pull/67509) (fix) |
| `hunyuan_a13b` | Text after `</answer>` is dropped | `R1_suffix` | not reported yet |

The two fixes linked above remove their findings. On the fix branches, `I1_flag` and `I2_text` drop to 0 for `deepseek_v31` and `hy_v3`.

## Truncated tool calls

When `max_tokens` cuts an output in the middle of a tool call, no complete call is left, and vLLM's parsers disagree on what non-streaming returns as `content`. The plugin records this as `policy_*` rows instead of failures. Each count below is a number of cut positions on `191e5d5`:

| `--tool-call-parser` | keeps the cut output | keeps only the text before the call | returns no content |
|---|---|---|---|
| `xlam`, `lfm2`, `longcat`, `jamba`, `pythonic`, `olmo3`, `phi4_mini_json`, `granite-20b-fc`, `granite4`, `hermes`, `step3`, `internlm` | all | 0 | 0 |
| `llama4_pythonic` | 113 | 0 | 0 |
| `gigachat3` | 182 | 3 | 3 |
| `kimi_k2` | 54 | 36 | 0 |
| `qwen3_coder`, `qwen3_xml` | 24 | 12 | 5 |
| `step3p5` | 24 | 15 | 5 |
| `granite` | 24 | 0 | 33 |
| `glm45`, `glm47` | 4 | 3 | 14 |
| `hy_v3` | 2 | 6 | 14 |
| `deepseek_v31` | 5 | 18 | 26 |
| `ernie45` | 9 | 0 | 69 |
| `deepseek_v3` | 14 | 36 | 131 |

Cuts that match none of the three cases (`policy_other`) are not counted. The question of which behavior is intended is open as [vllm#67513](https://github.com/vllm-project/vllm/issues/67513).

## Comparing with vLLM's Rust frontend

vLLM is porting its parsers to a Rust frontend in `rust/src/parser`. [`rust/parser_diff.rs`](rust/parser_diff.rs) is a small program for vLLM's `rust/src/chat` crate. It reads the recorded outputs and runs the Rust parser of the same name on each one. It feeds the text through the tokenizer and the incremental decoder, as the frontend does. `parser_conformance.rust_diff` then compares the Rust results with the Python ones.

```bash
cp rust/parser_diff.rs /path/to/vllm/rust/src/chat/examples/
(cd /path/to/vllm/rust && cargo build -p vllm-chat --example parser_diff)
# from the vLLM checkout, record the outputs with the Python results
PYTHONPATH=$PWD:/path/to/parser-conformance PC_REPORT=/tmp/pc.jsonl PC_SEEDS_OUT=/tmp/seeds.jsonl     python -m pytest tests/tool_parsers -n 8 -p parser_conformance.plugin -q
python -m parser_conformance.rust_diff rust/target/debug/examples/parser_diff /tmp/seeds.jsonl*
```

On `191e5d5`, in one run with `-n 8`, 54 recorded outputs have a parser in both implementations and a `tokenizer.json` in the local cache. 39 of them parse the same. All 15 differences come from three choices on the Rust side, each deliberate:

- When a cut leaves an unfinished call, the Rust parser's `finish()` reports it, and the frontend drops the unfinished markup. The Python parsers of the same names return the cut output as content. See [vllm#67513](https://github.com/vllm-project/vllm/issues/67513).
- The Rust DeepSeek V3 parser ignores text after `<｜tool▁calls▁end｜>`, and its tests assert this. Python keeps that text.
- Rust's `glm45` expects a newline after the function name. In Python, `glm45` is another name for the GLM-4.7 parser, which also accepts output without it.

The export builds a fresh Python parser for each output, and it sends the tool schemas that engine parsers take in `__init__`. Without these, Python and Rust disagree on outputs that both parse the same way.

`S_state` reports one parser on `191e5d5`: after a cut output, `granite4` leaves `in_tc` set, and the same instance then returns the next complete call as content. vLLM builds a new parser for each choice, so served requests do not reach this.

## Streaming cost

Set `PC_PERF=1` to time streaming as an output grows. For each tool parser, the plugin takes its longest recorded output and puts 256 or 1024 tokens of plain text into the first JSON string value, or in front of the output when it has none. A fresh parser then streams each version token by token. A time ratio near 4 means constant work per token. A ratio near 16 means every token rescans the whole text.

```bash
PYTHONPATH=$PWD:/path/to/parser-conformance PC_PERF=1 PC_PERF_OUT=/tmp/perf.jsonl PC_MAX_CUTS=1     python -m pytest tests/tool_parsers -n 8 -p parser_conformance.plugin -q
```

On `191e5d5`, 22 parsers give ratios from 1.0 to 4.8. `xlam` gives 8.1, from 22 ms to 177 ms: its streaming path runs regular expressions and `json.loads` over the whole text for every token. `ernie45` fails the streaming harness's own assertion, so it has no time.

## Checking the tool

The bugs above serve as controls. On a commit that has a bug, its row must show findings. On a commit with the fix, the row must show none:

```bash
git switch fix-deepseek-v31-unparsed-tool-calls   # branch of vllm#67491
PYTHONPATH=$PWD:/path/to/parser-conformance PC_REPORT=/tmp/neg.jsonl \
    python -m pytest tests/tool_parsers/test_deepseekv31_tool_parser.py -p parser_conformance.plugin -q
python -m parser_conformance.summarize /tmp/neg.jsonl   # no I1_flag or I2_text columns
```

## Limitations

- Coverage follows vLLM's tests. A parser with no test is not checked. When a test builds the parser with a stand-in tokenizer, such as a `MagicMock` with an empty vocabulary, streaming cannot see the marker token ids, so the plugin runs only the full-output and prefix checks and reports `skip_mock_tokenizer`. On `191e5d5`, 16 tool-parser test files do this.
- A finding is a candidate, not a verdict. Some differences are documented design:
  - `minicpm5_xml` drops content on purpose when it parses a call, and its tests assert this.
  - The `gemma4` reasoning tests expect different results per mode for output without a start marker.
  - Incremental streaming cannot take back a call it has started. JSON that never closes therefore becomes a call only in streaming, which the plugin reports as `I3_lenient`.
- `I2_text` ignores special tokens, `*_token` attributes of the parser, and tag-like strings such as `<answer>`. It does not know marker phrases such as Granite's `Here is my thought process:`, which is why the reasoning checks only compare a parser with itself.
- The streaming state is compared without an end-of-stream flush. Most `R2_trunc` findings are a parser holding back the start of a possible marker at the cut, for example `<th` or Granite's `Here`, which non-streaming returns as content.

## License

Apache-2.0, the same as vLLM. See [LICENSE](LICENSE).
