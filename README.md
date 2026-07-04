*This project has been created as part of the 42 curriculum by ayfadli.*

# Call Me Maybe
**An LLM function-calling tool powered by constrained decoding**

## Description
Small language models are often unreliable at producing structured output such as JSON. They can add commentary, break syntax, or hallucinate keys when asked to generate a function call.

Call Me Maybe solves that problem by constraining generation token by token. The project loads function schemas, guides the model with a structured prompt, and uses a custom decoding filter to keep output aligned with the expected JSON shape, ensuring near-perfect reliability even with a small 0.6B parameter model.

## Instructions

### Prerequisites
* Python 3.10 or later
* `uv` (for package and environment management)

### Installation
Install the project dependencies with one of the following commands:

```bash
make install
```

```bash
uv sync
```

The core project depends on `numpy` and `pydantic`. The provided `llm_sdk` uses `torch` and `transformers` internally to interact with the LLM.

### Execution
Run the main module with the default input and output directories:

```bash
uv run python -m src
```
By default, the program will read input files from the `data/input/` directory and write output to the `data/output/` directory.

You can also override the input, function-definition, and output paths:

```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

## Example Usage
Typical workflow:

```bash
# 1. Install dependencies
make install

# 2. Run the program with default paths
uv run python -m src

# 3. Check the structured output
cat data/output/function_calling_results.json
```

## Algorithm Explanation
The main decoding logic lives in `src/vocab_parser.py`. Instead of letting the model freely generate text, the program masks token probabilities and only keeps tokens that fit the current point in the JSON structure.

The process is split into phases:
1. The model is forced to emit the fixed prefix `{"name":"`.
2. Only function names from the input schema are allowed while the name is being generated.
3. After the function name, the decoder forces the bridge `","parameters":{`.
4. The parameter phase restricts tokens to allowable types (string, number, boolean) according to the schema constraints.
5. Generation stops once the closing braces are produced or the token limit is reached.
6. The modified logits ensure the chosen token maintains structural and semantic JSON validity by setting invalid tokens' logits to negative infinity before sampling.

## Design Decisions
* **CLI and I/O Parsing**: `src/__main__.py` handles CLI parsing, file operations, schema validation, and result serialization.
* **Constrained Decoding**: `src/vocab_parser.py` focuses entirely on filtering tokens based on the current JSON tree state.
* **Prompt Injection**: Function definitions are loaded from JSON and injected into the initial prompt context.
* **Validation**: Pydantic models validate both the expected input schemas and the final extracted JSON results, strictly adhering to the specified types.

## Performance Analysis
* **Accuracy**: Constrained decoding guarantees 100% syntactically valid JSON and schema-compliant outputs, yielding 90%+ correct function selection and argument extraction.
* **Speed**: By aggressively narrowing the search space through negative infinity logits for invalid tokens, the decoder maintains reasonable processing speeds. The program can process all test prompts in under 5 minutes on standard hardware.
* **Reliability**: Invalid token paths are pruned before sampling. The output is further verified to ensure it can be safely ingested by other systems.

## Challenges Faced
* **Tokenization Idiosyncrasies**: Tokenizers preserve whitespaces and punctuation in surprising ways. Finding exact prefix matches required careful alignment between the current generated string and token string representations.
* **JSON Syntax Overlaps**: Tokens may contain multiple structural characters (e.g., `": "`). The logic required strict prefix checks to guarantee the model didn't skip necessary JSON keys.
* **Free-Form Parameters**: While schema keys are strict, parameters are free-form strings, numbers, or booleans. Safely broadening the allowed token mask during parameter extraction without breaking JSON structure was a major technical hurdle.

## Testing Strategy
The implementation was robustly tested against:
* Assorted argument types: string parameters with spaces, numeric arguments, and booleans.
* Various edge cases: empty strings, large numbers, and special characters inside parameters.
* Functions requiring multiple parameters.
* Invalid, malformed, or ambiguous prompts, verifying that the program gracefully handles errors without crashing and provides clear error messages.
* Edge cases involving missing input files or malformed input JSON.

## Resources
* [Pydantic documentation](https://docs.pydantic.dev/)
* [NumPy documentation](https://numpy.org/doc/)
* JSON and constrained decoding concepts drawn from the project subject and related LLM function-calling materials.
* The `llm_sdk` module to interface directly with the Qwen3-0.6B LLM via `get_logits_from_input_ids`.

AI was used as a writing and review aid for documentation, especially to check wording, structure, and completeness against the subject requirements.