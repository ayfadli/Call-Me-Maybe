import string
import json
import numpy as np
from typing import Any


def get_allowed_chars(current_string: str, allowed_fn_names: list[str]) -> list[str]:
    target_prefix = '{"name":"'
    bridge = '","parameters":{'

    if (len(current_string) < len(target_prefix)):
        return [target_prefix[len(current_string):]]

    after_prefix = current_string[len(target_prefix):]

    # If no '"' in the after_prefix then, we don't have the function name yet
    if '"' not in after_prefix:
        allowed = []
        for name in allowed_fn_names:
            if name.startswith(after_prefix):
                remaining_string = name[len(after_prefix):] + '"'
                allowed.append(remaining_string)

        return allowed

    function_name = after_prefix.split('"')[0]
    perfect_string = target_prefix + function_name + bridge

    if len(current_string) < len(perfect_string):
        return [perfect_string[len(current_string):]]

    return list(string.printable)


def generate_constrained_json(
    model: Any,
    prompt_text: str,
    my_dict: dict[int, str],
    allowed_fn_names: list[str],
    raw_functions: list[dict[str, Any]],
    phase_4_valid_ids: list[int],
    clean_dict_items: list[tuple[int, str]],
) -> str:

    schema_hints = json.dumps(raw_functions)
    prompt = f"System: You are a strict API. Output ONLY valid JSON matching these schemas: {schema_hints}. CRITICAL: If a parameter is a 'number', output float format without quotes (e.g. 42.0, NOT \"42\"), apply this to all parameters.\nUser: {prompt_text}\nTool Call: "
    input_ids: list[int] = model.encode(prompt).tolist()[0]

    current_string: str = ""
    max_tokens: int = 150
    token_count: int = 0

    clean_dict_items = [(k, v) for k, v in my_dict.items() if v]

    target_phrases = allowed_fn_names + ['{"name":"', '","parameters":{']

    mini_dict = []
    # We scan the 150,000 tokens ONCE.
    # If a token is a substring of any of our target phrases, we keep it.
    for token_id, token_str in clean_dict_items:
        if any(token_str in phrase for phrase in target_phrases):
            mini_dict.append((token_id, token_str))

    while '}}' not in current_string.replace(" ", "").replace("\n", "") and token_count < max_tokens:

        allowed_rules: list[str] = get_allowed_chars(current_string, allowed_fn_names)

        logits: np.ndarray = np.array(model.get_logits_from_input_ids(input_ids))

        masked_logits: np.ndarray = np.full_like(logits, -np.inf)

        if (len(allowed_rules) > 10):
            # Phase 4 (Parameters): Use the massive VIP list passed from __main__
            valid_ids = phase_4_valid_ids

        else:
            valid_ids = [
                token_id for token_id, token_str in mini_dict
                if any(rule.startswith(token_str) for rule in allowed_rules)
            ]

        masked_logits[valid_ids] = logits[valid_ids]

        best_score: int = int(np.argmax(masked_logits))
        winning_string: str = my_dict.get(best_score)

        current_string += winning_string
        input_ids.append(best_score)

        token_count += 1
        print(f"\rGenerating: {current_string}", end="", flush=True)

    print()

    last_brace_idx = current_string.rfind('}')

    if last_brace_idx != -1:
        current_string = current_string[:last_brace_idx + 1]

    return current_string

