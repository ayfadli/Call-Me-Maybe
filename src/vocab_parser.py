import string
import re
import numpy as np
from typing import Any, Dict
import json
import time


def get_probabilities(logits: np.ndarray) -> np.ndarray:
    """Converts raw logits to a probability distribution."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def render_dashboard(
        model: Any, masked_logits: np.ndarray,
        top_3_raw_tokens: list[str],
        raw_scores: list[str], current_str: str) -> None:

    # THE FIX: [-3:][::-1] forces the array to be [Highest, Middle, Lowest]
    top_3_masked_ids = np.argsort(masked_logits)[-3:][::-1]
    top_3_masked_tokens = [model.decode([t]) for t in top_3_masked_ids]

    ai_wants = " | ".join(
        [f"'{repr(t)[1:-1]}' ({s})"
         for t, s in zip(top_3_raw_tokens, raw_scores)])
    we_got = repr(top_3_masked_tokens[0])[1:-1]

    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    CLEAR_LINE = '\033[K'

    # Now it correctly compares Rank 1 vs Rank 1
    if top_3_raw_tokens[0] != top_3_masked_tokens[0]:
        action = f"{RED}[FSM OVERRIDE]{RESET} -> Blocked "
        "'{repr(top_3_raw_tokens[0])[1:-1]}', Forced {GREEN}'{we_got}'{RESET}"
    else:
        action = f"{GREEN}[NATURAL]{RESET}      -> Allowed '{we_got}'"

    print(f"{CLEAR_LINE} {YELLOW}AI Brain  :{RESET} {ai_wants}")
    print(f"{CLEAR_LINE} {CYAN}FSM Engine:{RESET} {action}")
    print(f"{CLEAR_LINE} {GREEN}JSON Build:{RESET} {current_str}")
    print("\033[3A", end="", flush=True)


def get_allowed_chars(current_str: str, allowed_names: list[str]) -> list[str]:
    # Phase 1
    prefix = '{"name":"'
    if len(current_str) < len(prefix):
        return [prefix[len(current_str):]]

    # Phase 2: The function name
    after_prefix = current_str[len(prefix):]
    if '"' not in after_prefix:
        return [name[len(after_prefix):] + '"'
                for name in allowed_names if name.startswith(after_prefix)]

    # Phase 3: The bridge
    func_name = after_prefix.split('"')[0]
    target = prefix + func_name + '","parameters":{'
    if len(current_str) < len(target):
        return [target[len(current_str):]]

    # Phase 4: The arguments
    return list(string.printable)

def generate_constrained_json(
        model: Any, prompt_text: str, vocab_dict: Dict,
        allowed_fn: list[str], raw_functions: list[Dict[str, Any]],
        p4_valid_ids: list[int], clean_dict_items: list[tuple[int, str]],
        func_params: Dict[str, int], visualize: bool, param_types: Dict[str, Dict[str, Any]]) -> str:

    schema_hints = json.dumps(raw_functions)
    prompt = (
        "System: You are a strict API. Output only valid JSON matching "
        f"these schemas: {schema_hints}. CRITICAL: If a parameter is a "
        "'number', output float format without quotes (Example: if the parameter is an str output 'null').\n"
        f"User: {prompt_text}\n"
        "Tool Call: "
    )

    input_ids = model.encode(prompt).tolist()[0]
    vocab_size = np.array(model.get_logits_from_input_ids(input_ids)).shape[-1]

    target_phrases = allowed_fn + ['{"name":"', '","parameters":{', '}']
    mini_dict = [(i, s) for i, s in clean_dict_items if any(
        s in phrase for phrase in target_phrases)]

    p4_mask = np.zeros(vocab_size, dtype=bool)
    p4_mask[p4_valid_ids] = True

    p4_numbers_only = p4_mask.copy()
    for i, s in clean_dict_items:
        if s in string.digits:
            p4_numbers_only[i] = False

    p4_no_comma = p4_mask.copy()
    for i, s in clean_dict_items:
        if ',' in s:
            p4_no_comma[i] = False

    current_str = ""

    prefix = '{"name":"'
    current_str = prefix

    input_ids.extend(model.encode(prefix).tolist()[0])
    bridge_injected = False

    max_tokens = 150
    ids_blacklist = []
    is_recovering = False

    type_of_arg = "Any"
    while ('}}' not in current_str.replace(" ", "").replace("\n", "")
           and len(input_ids) < len(prompt) + max_tokens):

        rules = get_allowed_chars(current_str, allowed_fn)
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        mask = np.zeros(vocab_size, dtype=bool)

        if visualize:
            # np.argsort sorts smallest to largest.
            # [-3:] gets the last 3 (the highest).
            top_3_ids = np.argsort(logits)[-3:][::-1]
            top_3_tokens = [model.decode([t]) for t in top_3_ids]
            # scores = [round(logits[t], 2) for t in top_3_ids]

        if len(rules) > 10:
            # Phase 4: Quota Shield
            clean_curr = current_str.replace(" ", "")

            func_name = clean_curr.split('"name":"')[1].split(
                '"')[0] if '"name":"' in current_str else ""

            params = clean_curr.split('"parameters"')[
                1] if '"parameters"' in current_str else ""

            # Enter if we are in the last parmeter
            active_keys = re.findall(r'"([^"]+)"\s*:\s*$', params, re.MULTILINE)
            print(active_keys)
            if len(active_keys) > 0:
                active_keys = active_keys[-1:]
                type_of_arg = param_types[func_name].get(active_keys[0], []) if len(active_keys) > 0 else []

            print(f"type of arg {type_of_arg}")
            param_count = len(re.findall(r'"([^"]+)"\s*:', params))
            if (param_count == func_params.get(func_name, 99)):
                if params.count('"') % 2 != 0:
                    mask = p4_mask  # Inside string value
                # If the last parameter the AI generated was a String.
                elif current_str.strip().endswith('"'):
                    for i, s in mini_dict:
                        # Force the AI to close the JSON
                        if s.strip() == '}':
                            mask[i] = True
                else:
                    mask = p4_no_comma

            elif type_of_arg == "number":
                mask = p4_numbers_only

            else:
                # Target Quota is NOT met yet. Keep generating parameters.
                mask = p4_mask

        # Phase 1-3: Strict Spelling
        else:
            for i, s in mini_dict:
                if any(rule.startswith(s) for rule in rules):
                    mask[i] = True

        logits[~mask] = -np.inf

        # The recovery system
        if np.max(logits) == -np.inf:
            bad_token_id = input_ids.pop()
            current_str = model.decode(input_ids)
            ids_blacklist.append(bad_token_id)
            is_recovering = True
            print("Dead end detected! Initiating rollback...")
            continue

        if is_recovering:
            for bad_id in ids_blacklist:
                logits[bad_id] = -np.inf
        else:
            ids_blacklist.clear()

        is_recovering = False

        best_id = int(np.argmax(logits))
        current_str += vocab_dict.get(best_id, "")
        input_ids.append(best_id)

        if current_str.endswith('"') and not bridge_injected and prefix in current_str:
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(model.encode(bridge).tolist()[0])
            bridge_injected = True
            continue

        if visualize:
            top_3_ids = np.argsort(logits)[-3:][::-1]
            top_3_tokens = [model.decode([t]) for t in top_3_ids]

            probs = get_probabilities(logits)
            scores = [f"{probs[t] * 100:.1f}%" for t in top_3_ids]
            render_dashboard(model, logits, top_3_tokens, scores, current_str)

            time.sleep(1)
        else:
            print(f"\rGenerating: {current_str}", end="", flush=True)

    if visualize:
        print("\n\n\n")

    print()
    return current_str[:current_str.rfind('}') + 1] if '}' in current_str else current_str
