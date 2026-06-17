import argparse, json, sys
from pydantic import BaseModel, ValidationError
from typing import Any, Dict
import pathlib
from llm_sdk import Small_LLM_Model
from src.vocab_parser import run_bouncer
from datetime import datetime
import string
import re

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]


def parse_arguments() -> argparse.Namespace:
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
                        default="data/output/function_calling_results.json",
                        help="Path to the JSON output file.")



    args = parser.parse_args()
    return args

def load_json_file(filename: str) -> Any:

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



def main() -> None:
    print(f"Start: {datetime.now().time()}")
    args = parse_arguments()

    raw_functions: list[dict[str, Any]] = load_json_file(args.functions_definition)
    raw_prompts: list[dict[str, Any]] = load_json_file(args.input)

    allowed_fn_names: list[str] = []

    model = Small_LLM_Model()
    vocab_file = model.get_path_to_vocab_file()
    my_dict: dict[int, str] = load_json_file(vocab_file)

    my_dict = {v: k.replace('Ġ', ' ') for k, v in my_dict.items()}

    printable_set: set[str] = set(string.printable)

    #This is a list comprehension to filter the tokens (words or chars), to keep just the valid tokens (ids).
    valid_ids: list[int] = [
        token_id for token_id, token_str in my_dict.items()
        if token_str and all(c in printable_set for c in token_str)
    ]

    # 2. Build a globally filtered dictionary (strips out ~120,000 useless foreign tokens!)
    clean_dict_items: list[tuple[int, str]] = [
        (k, v) for k, v in my_dict.items()
        if v and all(c in printable_set for c in v)
    ]

    for fn in raw_functions:
        try:
            func = FunctionDef(**fn)
            print(fn)
            allowed_fn_names.append(func.name)

        except ValidationError as e:
            print("Validation failed: input data is invalid or incomplete.", file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)

        except Exception as e:
            print(f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    final_results_list: list[dict[str, Any]] = []
    for prompt in raw_prompts:
        prompt_text: str = prompt['prompt']

        print(f"\nPrompt: {prompt_text}")

        raw_json_string = run_bouncer(
            model,
            prompt_text,
            my_dict,
            allowed_fn_names,
            raw_functions,
            valid_ids,
            clean_dict_items
        )

        try:

            #The (?!...) part means "not followed by"
            json_str = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', raw_json_string)
            extracted_dict = json.loads(json_str)

            for func in raw_functions:
                if func['name'] == extracted_dict['name']:
                    expected_params = func.get("parameters", {})

                    for key, val in extracted_dict["parameters"].items():

                        if key in expected_params and expected_params[key].get("type") == "number":
                            if isinstance(val, int) and not isinstance(val, bool):
                                extracted_dict["parameters"][key] = float(val)
                        break

            final_data = {
                "prompt": prompt_text,
                "name": extracted_dict["name"],
                "parameters": extracted_dict["parameters"]
            }

            result = FunctionCallResult(**final_data)
            final_results_list.append(result.model_dump())

        except ValidationError as e:
            print("Validation failed: output data is invalid or incomplete.", file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)

        except Exception as e:
            print(f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    if not pathlib.Path(args.output).is_file():
        print(f"This file '{args.output}' is not found, or it is a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.output, 'w') as file:
            json.dump(final_results_list, file, indent=4)

    except PermissionError:
        print(f"Permission denied in this file {args.output}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

    print("All done !")
    print(f"End of program: {datetime.now().time()}")
if __name__ == "__main__":
    main()
