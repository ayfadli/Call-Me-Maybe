import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from llm_sdk import load_model, generate_function_call, extract_json_from_output, FunctionDef, FunctionCallResult

def parse_arguments():
    parser = argparse.ArgumentParser(description="Call Me Maybe: LLM Function Calling Tool")

    parser.add_argument(
        "--functions_definitions",
        type=str,
        default="data/input/function_definitions.json",
        help="Path to the JSON file containing function schemas."
    )

    parser.add_argument(
        "--input",
        "-I",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing a simple array of the natural language prompts that your system must process."
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the JSON file that should containing an array of generated objects."
    )
    args = parser.parse_args()
    return args

def load_json_file(filepath: str) -> list | dict:
    if not Path(filepath).is_file():
        print("Error: File not found")
        sys.exit(1)
    try:
        with open(filepath, 'r') as file:
            return json.load(file)

    except json.JSONDecodeError as e:
        print(f"Error: '{filepath}' contains invalid JSON (corrupted or missing a comma).\nDetails: {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: You do not have permission to read '{filepath}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An enxepected error occured while reading: '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    args = parse_arguments()

    definitions = args.functions_definitions
    input_file = args.input
    output_file =  args.output

    raw_functions = load_json_file(definitions)
    raw_prompts = load_json_file(input_file)

    available_functions = []
    try:
        for item in raw_functions:
            available_functions.append(FunctionDef(**item))
        print(f"Successfuly loaded and validated {len(available_functions)} functions.")
    except ValidationError as e:
        print(f"Error: The shema is invalid (missing key or wrong data type).\n'{e}'", file=sys.stderr)
        sys.exit(1)

    model, tokenizer = load_model()

    print("\nFiring up the inference engine...\n")
    print("-" * 50)

    final_results_list = []

    for test in raw_prompts:
        prompt_text = test['prompt']
        print(f"User Prompt: {prompt_text}")
        print("Thinking...")

        raw_response = generate_function_call(model, tokenizer, prompt_text, available_functions)

        print(f"AI Output:\n{raw_response}")
        print("-" * 50)

        extracted_dict = extract_json_from_output(raw_response)

        if extracted_dict:
            final_data = {
                "prompt": prompt_text,
                "name": extracted_dict["name"],
                "parameters": extracted_dict["arguments"]
            }

            try:
                validated_result = FunctionCallResult(**final_data)

                final_results_list.append(validated_result.model_dump())
                print("Successfully parsed and validated!")

            except ValidationError as e:
                print(f"Pydantic Validation Failed: {e}")

    print("-" * 50)
    print("Saving results to disk...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results_list, f, indent=4)

    print(f"SUCCESS! Saved {len(final_results_list)} results to {output_file}.")

if __name__ == "__main__":
    main()
