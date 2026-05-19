*This project has been created as part of the 42 curriculum by ayfadli.*

# Call Me Maybe
**An LLM function-calling tool powered by constrained decoding**

## Description
Small language models are often unreliable at producing structured output such as JSON. They can add commentary, break syntax, or hallucinate keys when asked to generate a function call.

Call Me Maybe solves that problem by constraining generation token by token. The project loads function schemas, guides the model with a structured prompt, and uses a custom decoding filter to keep output aligned with the expected JSON shape.

## Instructions

### Prerequisites
* Python 3.10 or later
* `uv`

### Installation
Install the project dependencies with one of the following commands:

```bash
make install
```

```bash
uv sync
```

The project depends on `accelerate`, `numpy`, `pydantic`, `torch`, and `transformers`.

### Execution
Run the main module with the default input files:

```bash
uv run python -m src
```

You can also override the input, function-definition, and output paths:

```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calling_results.json
```

The current implementation expects the output file to exist before execution. If needed, create it first:

```bash
mkdir -p data/output
touch data/output/function_calling_results.json
```

## Example Usage
Typical workflow:

```bash
make install
touch data/output/function_calling_results.json
uv run python -m src
cat data/output/function_calling_results.json
```

## Algorithm Explanation
The main decoding logic lives in `src/vocab_parser.py`. Instead of letting the model freely generate text, the program masks token probabilities and only keeps tokens that fit the current point in the JSON structure.

The process is split into phases:

1. The model is forced to emit the fixed prefix `{"name":"`.
2. Only function names from the input schema are allowed while the name is being generated.
3. After the function name, the decoder forces the bridge `","parameters":{`.
4. The parameter phase allows printable tokens so the model can fill in arguments.
5. Generation stops once the closing braces are produced or the token limit is reached.

## Design Decisions
* `src/__main__.py` handles CLI parsing, schema validation, file I/O, and result serialization.
* `src/vocab_parser.py` focuses on constrained decoding and token filtering.
* Function definitions are loaded from JSON and injected into the prompt so the model knows which tool names are valid.
* Pydantic models are used to validate both the input schemas and the final extracted results.

## Performance Analysis
* Accuracy is improved because invalid token paths are removed before sampling.
* Speed stays reasonable because the decoder narrows the search space aggressively.
* Reliability improves because the output is validated before it is written to disk.

## Challenges Faced
* Tokenizers can merge punctuation and spaces in surprising ways, so the decoder must compare exact token strings carefully.
* Some tokens can try to jump ahead in the JSON structure, which requires strict prefix checks.
* Free-form parameter generation needs a broad enough printable-character set to avoid dead ends.

## Testing Strategy
The project was tested with prompts that cover:

* String parameters with spaces and punctuation.
* Numeric arguments that must stay unquoted.
* Functions with multiple parameters.
* Invalid or ambiguous prompts to verify validation and error handling.

## Resources
* [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
* [PyTorch documentation](https://pytorch.org/docs/)
* [Pydantic documentation](https://docs.pydantic.dev/)
* JSON and constrained decoding concepts from the project subject and related LLM function-calling material

AI was used as a writing and review aid for documentation, especially to check wording, structure, and completeness against the subject requirements. The implementation itself was designed to be understood and maintained manually, with the generated output validated against the project rules.
