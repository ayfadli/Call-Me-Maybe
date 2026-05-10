import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError, BaseModel
from typing import Dict, Any
from llm_sdk import Small_LLM_Model
from src.vocab_parser import run_bouncer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Call Me Maybe: LLM Function Calling Tool")

    parser.add_argument(
        "--functions_definitions",
        type=str,
        default="data/input/functions_definition.json",
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


class FunctionDef(BaseModel):
    name:str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def load_json_file(filepath: str) -> list | dict:
    if not Path(filepath).is_file():
        print(f"Error: '{filepath}' File not found")
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

    raw_functions = load_json_file(args.functions_definitions)
    raw_prompts = load_json_file(args.input)

    # 1. Load available functions and extract their names
    available_functions = []
    try:
        for item in raw_functions:
            available_functions.append(FunctionDef(**item))
        print(f"Successfully loaded {len(available_functions)} functions.")
    except ValidationError as e:
        print(f"Schema error: {e}", file=sys.stderr)
        sys.exit(1)

    allowed_fn_names = [fn.name for fn in available_functions]

    # 2. Boot the AI and load the Vocabulary dictionary ONCE
    print("\nBooting AI and loading vocabulary...")
    model = Small_LLM_Model()

    vocab_file = model.get_path_to_vocab_file()
    with open(vocab_file, 'r') as v:
        my_dict = json.load(v)
    my_dict = {v: k.replace('Ġ', ' ') for k, v in my_dict.items()}

    print("-" * 50)
    final_results_list = []

    # 3. The Generation Loop
    for test in raw_prompts:
        prompt_text = test['prompt']
        print(f"\nUser Prompt: {prompt_text}")

        # Let the Bouncer do the heavy lifting!
        raw_json_string = run_bouncer(model, prompt_text, my_dict, allowed_fn_names, raw_functions)

        # Parse the perfect string into a real Python dictionary
        try:
            extracted_dict = json.loads(raw_json_string)

            # Match the Pydantic schema
            final_data = {
                "prompt": prompt_text,
                "name": extracted_dict["name"],
                "parameters": extracted_dict["parameters"]
            }

            validated_result = FunctionCallResult(**final_data)
            final_results_list.append(validated_result.model_dump())
            print("✅ Successfully parsed and validated!")

        except json.JSONDecodeError:
            print("❌ Failed to parse JSON. (This should rarely happen now!)")
        except ValidationError as e:
            print(f"❌ Pydantic Validation Failed: {e}")

    # 4. Save to Disk
    print("\n" + "-" * 50)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results_list, f, indent=4)

    print(f"🎉 SUCCESS! Saved {len(final_results_list)} results to {args.output}.")

if __name__ == "__main__":
    main()
