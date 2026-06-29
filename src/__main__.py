import argparse
import json
import sys
from pydantic import BaseModel, ValidationError
from typing import Any, Dict
import pathlib
from llm_sdk import Small_LLM_Model
from src.vocab_parser import generate_constrained_json
from datetime import datetime
import string
import re
from rich import print_json
from dataclasses import dataclass
import numpy as np
from typing import Any, Dict, List, Tuple


@dataclass
class MaskCache:
    model: Any
    vocab_dict: Dict[int, str]

    allowed_fn: List[str]
    raw_functions: List[Dict[str, Any]]  # Needed for schema_hints
    func_params: Dict[str, int]
    param_types: Dict[str, Dict[str, Any]]

    p4_mask: np.ndarray
    p4_numbers_only: np.ndarray
    p4_no_comma: np.ndarray
    mini_dict: List[Tuple[int, str]]
    clean_dict_items: List[Tuple[int, str]] # Keep this if you still loop over it in Rule A

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
        usage="uv run python -m src [--functions_definition "
        "<function_definition_file>] [--input <input_file>] [--output <output_file>]"
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

    parser.add_argument('--model',
                        type=str,
                        default="Qwen/Qwen3-0.6B",
                        help="HuggingFace Model ID")

    args = parser.parse_args()
    return args


def load_json_file(filename: str) -> Any:

    if not pathlib.Path(filename).is_file():
        raise SystemExit(
            f"This file {filename} is not found, or it is a directory.")

    try:
        with open(filename, 'r') as file:
            json_data = json.load(file)

    except json.JSONDecodeError:
        print("The JSON file is invalid or corrupted. "
              "A required key or value is missing.", file=sys.stderr)
        sys.exit(1)

    except PermissionError:
        print(f"Permission denied in this file {filename}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"An error occured.\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

    return json_data


def main() -> None:
    args = parse_arguments()

    raw_functions: list[dict[str, Any]] = load_json_file(
        args.functions_definition)
    raw_prompts: list[dict[str, Any]] = load_json_file(args.input)

    allowed_fn_names: list[str] = []

    model = Small_LLM_Model(model_name=args.model)

    start_time = datetime.now()
    vocab_file = model.get_path_to_vocab_file()
    vocab_dict: dict[int, str] = load_json_file(vocab_file)

    vocab_dict = {v: k.replace('Ġ', ' ') for k, v in vocab_dict.items()}

    printable_set: set[str] = set(string.printable)

    # Filter the tokens, to keep just the valid tokens (ids).
    valid_ids: list[int] = [
        token_id for token_id, token_str in vocab_dict.items()
        if token_str and all(c in printable_set for c in token_str)
    ]

    # 2. strips out ~120,000 useless foreign tokens!
    clean_dict_items: list[tuple[int, str]] = [
        (k, v) for k, v in vocab_dict.items()
        if v and all(c in printable_set for c in v)
    ]

    func_params = dict()
    param_types = dict()
    for fn in raw_functions:
        try:
            func = FunctionDef(**fn)
            allowed_fn_names.append(func.name)
            func_params[func.name] = len(func.parameters)

            params = fn.get("parameters", {})
            param_types[func.name] = {}

            for param_key, details in params.items():
                param_types[func.name][param_key] = details.get("type", string)

        except ValidationError as e:
            print(
                "Validation failed: input data is invalid or incomplete.",
                file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)

        except Exception as e:
            print(
                f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    # We need the vocab size to build the masks
    dummy_input_ids = model.encode("dummy").tolist()[0]
    vocab_size = len(model.get_logits_from_input_ids(dummy_input_ids))

    # Pre-calculate p4_mask
    p4_mask = np.zeros(vocab_size, dtype=bool)
    p4_mask[valid_ids] = True

    # Pre-calculate p4_numbers_only
    p4_numbers_only = np.zeros(vocab_size, dtype=bool)
    allowed_math_chars = set("0123456789.-, }")
    for i, s in clean_dict_items:
        if all(char in allowed_math_chars for char in s) or s == "null":
            p4_numbers_only[i] = True

    # Pre-calculate p4_no_comma
    p4_no_comma = p4_mask.copy()
    for i, s in clean_dict_items:
        if ',' in s:
            p4_no_comma[i] = False

    # Pre-calculate mini_dict
    target_phrases = allowed_fn_names + ['{"name":"', '","parameters":{', '}']
    mini_dict = [(i, s) for i, s in clean_dict_items if any(s in phrase for phrase in target_phrases)]

    # Bundle everything into the Cache object
    cache = MaskCache(
        model=model,
        vocab_dict=vocab_dict,
        allowed_fn=allowed_fn_names,
        raw_functions=raw_functions,
        func_params=func_params,
        param_types=param_types,
        p4_mask=p4_mask,
        p4_numbers_only=p4_numbers_only,
        p4_no_comma=p4_no_comma,
        mini_dict=mini_dict,
        clean_dict_items=clean_dict_items
    )

    final_results_list: list[dict[str, Any]] = []
    for prompt in raw_prompts:
        prompt_text: str = prompt['prompt']

        print(f"\nPrompt: {prompt_text}")

        raw_json_string = generate_constrained_json(prompt_text, cache)

        try:
            # The (?!...) part means "not followed by"
            json_str = re.sub(
                r'(?<!\\)\\(?![/"\\bfnrtu])', r'\\\\', raw_json_string)
            extracted_dict = json.loads(json_str)

            fn_name = extracted_dict.get("name")
            expected_params = {}
            for fn in raw_functions:
                if fn.get("name") == fn_name:
                    expected_params = fn.get("parameters", {})
                    break

            if "parameters" in extracted_dict:
                for key, val in extracted_dict["parameters"].items():
                    if (
                        key in expected_params
                        and expected_params[key].get("type") == "number"
                    ):
                        if isinstance(val, int) and not isinstance(val, bool):
                            extracted_dict["parameters"][key] = float(val)
                    elif isinstance(val, str):
                        extracted_dict["parameters"][key] = val.strip()

            final_data = {
                "prompt": prompt_text,
                "name": extracted_dict["name"],
                "parameters": extracted_dict["parameters"]
            }

            result = FunctionCallResult(**final_data)
            final_results_list.append(result.model_dump())

            print()
            print_json(data=final_data)

        except ValidationError as e:
            print(
                "Validation failed: output data is invalid or incomplete.",
                file=sys.stderr)
            print(e.errors()[0])
            sys.exit(1)

        except Exception as e:
            print(
                f"An unexpected error occured.\nDetails: {e}", file=sys.stderr)
            sys.exit(1)

    output_file = pathlib.Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    pathlib.Path(args.output).touch(exist_ok=True)

    if not output_file.is_file():
        print(
            f"This file '{args.output}' is not found, or it is a directory.",
            file=sys.stderr)
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

    elapsed_time = datetime.now() - start_time
    total_seconds = elapsed_time.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"\nAll done in {minutes}m {seconds}s!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
