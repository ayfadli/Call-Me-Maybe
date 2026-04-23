import argparse
import json
import sys
from pathlib import Path
from src.shema import FunctionDef
from pydantic import ValidationError

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


if __name__ == "__main__":
    main()
