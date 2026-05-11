import argparse, json, sys
from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import pathlib

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="CallMeMaybe",
        description="Call Me Maybe: LLM Function Calling Tool",
        usage="uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]",
        # epilog="options: \n--functions_definition \tPath to the JSON file containing the functions definitions.\n--input \tPath to the JSON file containing the prompts.\n--output \tPath to the JSON output file."
    )

    parser.add_argument(
                        '--functions_definition',
                        metavar='',
                        type=str,
                        default="data/input/functions_definition.json",
                        help="Path to JSON file containing the functions definitions."
                        )

    parser.add_argument('--input',
                        metavar='',
                        type=str,
                        default="data/input/function_calling_tests.json",
                        help="Path to the file containing the prompts.")

    parser.add_argument(
                        '--output',
                        metavar='',
                        type=str,
                        default="data/output/function_calls.json",
                        help="Path to the JSON output file.")



    args = parser.parse_args()
    return args

def load_json_file(filename):

    if not pathlib.Path(filename).is_file():
        print(f"This file {filename} is not found, or it is a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)

    except json.JSONDecodeError:
        print("The JSON file is invalid or corrupted. A required key or value is missing.", file=sys.stderr)
        sys.exit(1)

    except PermissionError:
        print(f"Permission denied in this file {filename}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

    return json_data



def main():
    args = parse_arguments()

    raw_functions = load_json_file(args.functions_definition)
    raw_prompts = load_json_file(args.input)

    allowed_fn_names = []

    for fn in raw_functions:
        try:
            func = FunctionDef(**fn)
            allowed_fn_names.append(func.name)

        except ValidationError as e:
            print("Validation failed: input data is invalid or incomplete.", file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)

        except Exception as e:
            print(f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    print(allowed_fn_names)

if __name__ == "__main__":
    main()
