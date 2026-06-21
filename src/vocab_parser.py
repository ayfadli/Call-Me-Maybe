import string
import json
import re
import numpy as np
from typing import Any

def get_allowed_chars(current_str: str, allowed_names: list[str]) -> list[str]:
    prefix = '{"name":"'
    if len(current_str) < len(prefix):
        return [prefix[len(current_str):]]

    after_prefix = current_str[len(prefix):]
    if '"' not in after_prefix:
        return [name[len(after_prefix):] + '"' for name in allowed_names if name.startswith(after_prefix)]

    func_name = after_prefix.split('"')[0]
    target = prefix + func_name + '","parameters":{'

    if len(current_str) < len(target):
        return [target[len(current_str):]]

    return list(string.printable)

def generate_constrained_json(
    model: Any, prompt_text: str, my_dict: dict[int, str], allowed_fn_names: list[str],
    raw_functions: list[dict[str, Any]], phase_4_valid_ids: list[int],
    clean_dict_items: list[tuple[int, str]], func_parameters: dict[str, int]
) -> str:

    schema = json.dumps(raw_functions)
    prompt = f"System: You are a strict API. Output ONLY valid JSON matching these schemas: {schema}. CRITICAL: If a parameter is a 'number', output float format without quotes.\nUser: {prompt_text}\nTool Call: "

    input_ids = model.encode(prompt).tolist()[0]
    vocab_size = np.array(model.get_logits_from_input_ids(input_ids)).shape[-1]

    target_phrases = allowed_fn_names + ['{"name":"', '","parameters":{', '}']
    mini_dict = [(i, s) for i, s in clean_dict_items if s.strip() == "}" or any(s in phrase for phrase in target_phrases)]

    p4_mask = np.zeros(vocab_size, dtype=bool)
    p4_mask[phase_4_valid_ids] = True

    p4_no_comma = p4_mask.copy()
    for i, s in clean_dict_items:
        if "," in s: p4_no_comma[i] = False

    current_str = ""

    prefix = '{"name":"'
    current_str = prefix

    input_ids.extend(model.encode(prefix).tolist()[0])
    bridge_injected = False

    max_tokens = 150

    while '}}' not in current_str.replace(" ", "").replace("\n", "") and len(input_ids) < len(prompt) + max_tokens:

        rules = get_allowed_chars(current_str, allowed_fn_names)
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        mask = np.zeros(vocab_size, dtype=bool)

        if len(rules) > 10:
            # Phase 4: Quota Shield
            clean_curr = current_str.replace(" ", "")
            func_name = clean_curr.split('"name":"')[1].split('"')[0] if '"name":"' in clean_curr else ""
            params = current_str.split('"parameters"')[1] if '"parameters"' in current_str else ""

            if len(re.findall(r'"([^"]+)"\s*:', params)) == func_parameters.get(func_name, 99):
                if params.count('"') % 2 != 0:
                    mask = p4_mask  # Inside string value
                elif current_str.strip().endswith('"'):
                    for i, s in mini_dict:
                        if s.strip() == "}": mask[i] = True  # Kill switch
                else:
                    mask = p4_no_comma  # Block commas
            else:
                mask = p4_mask
        else:
            # Phases 1-3: Strict Spelling
            for i, s in mini_dict:
                if any(r.startswith(s) for r in rules): mask[i] = True

        # Bare-metal mutation
        logits[~mask] = -np.inf
        best_id = int(np.argmax(logits))

        current_str += my_dict.get(best_id, "")
        input_ids.append(best_id)

        if current_str.endswith('"') and not bridge_injected and prefix in current_str:
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(model.encode(bridge).tolist()[0])
            bridge_injected = True
            continue

        print(f"\rGenerating: {current_str}", end="", flush=True)

    print()
    return current_str[:current_str.rfind('}') + 1] if '}' in current_str else current_str
