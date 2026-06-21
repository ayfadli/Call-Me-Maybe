import string
import json
import numpy as np
from typing import Any
import re


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
    func_parameters: dict[str, int]
) -> str:

    schema_hints = json.dumps(raw_functions)
    prompt = f"System: You are a strict API. Output ONLY valid JSON matching these schemas: {schema_hints}. CRITICAL: If a parameter is a 'number', output float format without quotes (e.g. 42.0, NOT \"42\"), apply this to all parameters.\nUser: {prompt_text}\nTool Call: "
    input_ids: list[int] = model.encode(prompt).tolist()[0]

    current_string: str = ""
    max_tokens: int = 150
    token_count: int = 0

    vocab_logits = np.array(model.get_logits_from_input_ids(input_ids))
    vocab_size = vocab_logits.shape[-1]

    target_phrases = allowed_fn_names + ['{"name":"', '","parameters":{', '}']
    mini_dict = []

    for token_id, token_str in clean_dict_items:
        if any(token_str in phrase for phrase in target_phrases) or token_str.strip() == "}":
            mini_dict.append((token_id, token_str))

    # Pre-build the Phase 4 Boolean Mask
    phase_4_mask = np.zeros(vocab_size, dtype=bool)
    for v_id in phase_4_valid_ids:
        phase_4_mask[v_id] = True

    # Pre-build the No-Comma Mask (for when the quota is met)
    phase_4_no_comma_mask = phase_4_mask.copy()
    for token_id, token_str in clean_dict_items:
        if "," in token_str:
            phase_4_no_comma_mask[token_id] = False
    # -----------------------------------------------------------

    while '}}' not in current_string.replace(" ", "").replace("\n", "") and token_count < max_tokens:

        allowed_rules: list[str] = get_allowed_chars(current_string, allowed_fn_names)

        # Get raw logits from the model
        logits: np.ndarray = np.array(model.get_logits_from_input_ids(input_ids))

        # We will use this active_mask to modify logits directly!
        active_mask = np.zeros(vocab_size, dtype=bool)

        if len(allowed_rules) > 10:
            # --- PHASE 4: PARAMETERS & QUOTA SHIELD ---
            func_name = ""
            params_section = ""
            found_keys = []

            # 1. Safely extract data if we are inside the parameters block
            if '"parameters":' in current_string.replace(" ", ""):
                try:
                    params_section = current_string.split('"parameters"')[1]
                    func_name = current_string.replace(" ", "").split('"name":"')[1].split('"')[0]
                except IndexError:
                    pass
                # Find all parameter keys the AI has typed so far
                found_keys = re.findall(r'"([^"]+)"\s*:', params_section)

            target_count = func_parameters.get(func_name, 99)
            parameters_written = len(found_keys)

            # 2. THE QUOTA SHIELD
            if parameters_written == target_count:
                # Count quotes to see if we are inside a string value
                quote_count = params_section.count('"')
                is_inside_string = (quote_count % 2 != 0)
                last_char = current_string.strip()[-1] if current_string.strip() else ""

                if is_inside_string:
                    # Inside a string: Let it type normally
                    active_mask = phase_4_mask
                elif last_char == '"':
                    # Just finished a string: THE KILL SWITCH
                    for token_id, token_str in mini_dict:
                        if token_str.strip() == "}":
                            active_mask[token_id] = True
                else:
                    # Typing a number/boolean: Use the No-Comma shield
                    active_mask = phase_4_no_comma_mask
            else:
                # Quota not met yet: Use the standard Phase 4 mask
                active_mask = phase_4_mask

        else:
            # --- PHASES 1, 2, & 3: STRICT SPELLING ---
            for token_id, token_str in mini_dict:
                if any(rule.startswith(token_str) for rule in allowed_rules):
                    active_mask[token_id] = True

        # --- ⚡ BARE-METAL MUTATION ⚡ ---
        # NO new arrays. NO copying. Just C-level math.
        # "Set all logits where the mask is False to negative infinity"
        logits[~active_mask] = -np.inf

        # Grab the winning token ID
        best_score: int = int(np.argmax(logits))
        winning_string: str = my_dict.get(best_score, "")

        current_string += winning_string
        input_ids.append(best_score)

        token_count += 1
        print(f"\rGenerating: {current_string}", end="", flush=True)

    print()

    # Clean up any trailing garbage after the final brace
    last_brace_idx = current_string.rfind('}')
    if last_brace_idx != -1:
        current_string = current_string[:last_brace_idx + 1]

    return current_string
