from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import json
import numpy as np
import string
import re


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

def generate_constrained_json(prompt_text: str, cache: Any) -> str:

    optimized_schemas = []
    for f in cache.raw_functions:
        optimized_schemas.append({
            "name": f["name"],
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {})
        })

    # Use separators to remove ALL spaces from the JSON string!
    schema_hints = json.dumps(optimized_schemas, separators=(',', ':'))

    prompt = (
        f"System: Output valid JSON matching these schemas: {schema_hints}\n"
        f"User: {prompt_text}\n"
        "Tool Call: "
    )

    input_ids = cache.model.encode(prompt).tolist()[0]
    vocab_size = len(cache.model.get_logits_from_input_ids(input_ids))

    current_str = ""

    prefix = '{"name":"'
    current_str = prefix

    input_ids.extend(cache.model.encode(prefix).tolist()[0])
    bridge_injected = False

    max_tokens = 150

    while ('}}' not in current_str.replace(" ", "").replace("\n", "")
           and len(input_ids) < len(prompt) + max_tokens):

        if prefix in current_str and '","parameters":{' not in current_str:
            after_prefix = current_str.split(prefix)[1]
            possible_names = [n for n in cache.allowed_fn if n.startswith(after_prefix)]

            # If the LLM has typed enough to narrow it down to exactly ONE function:
            if len(possible_names) == 1 and possible_names[0] != after_prefix:
                # Teleport to the end of the word!
                remainder = possible_names[0][len(after_prefix):] + '"'
                current_str += remainder
                input_ids.extend(cache.model.encode(remainder).tolist()[0])
                continue

        # 2. Bridge Fast-Forward (FIXED!)
        # We added: len(current_str) > len(prefix) so it doesn't trigger on loop 1
        if (current_str.endswith('"')
            and not bridge_injected
            and prefix in current_str
            and len(current_str) > len(prefix)):

            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(cache.model.encode(bridge).tolist()[0])
            bridge_injected = True

            # If the function takes ZERO parameters, end the JSON instantly!
            func_name = current_str.split('"name":"')[1].split('"')[0]
            if cache.func_params.get(func_name, 99) == 0:
                current_str += "}}"
                break
            active_schema = next((f for f in cache.raw_functions if f["name"] == func_name), None)

            if active_schema:
                tiny_schema = json.dumps(
                    [{
                        "name": active_schema["name"],
                        "description": active_schema.get("description", ""),
                        "parameters": active_schema.get("parameters", {})
                    }],
                    separators=(',', ':')
                )

                # RESTORED SYSTEM/USER KEYWORDS!
                tiny_prompt = (
                    f"System: Output valid JSON matching this schema: {tiny_schema}\n"
                    f"User: {prompt_text}\n"
                    f"Tool Call: {current_str}"
                )

                # OVERWRITE the input_ids with the new, tiny version!
                input_ids = cache.model.encode(tiny_prompt).tolist()[0]
            else:
                # Fallback just in case
                input_ids.extend(cache.model.encode(bridge).tolist()[0])
            # ===============================================================

            continue

        rules = get_allowed_chars(current_str, cache.allowed_fn)
        logits = np.array(cache.model.get_logits_from_input_ids(input_ids))
        mask = np.zeros(vocab_size, dtype=bool)

        if len(rules) > 10:
            # --- PHASE 4: THE QUOTA & TYPE SHIELD ---

            # 1. Safely extract function name
            clean_curr = current_str.replace(" ", "").replace("\n", "")
            func_name = clean_curr.split('"name":"')[1].split('"')[0] if '"name":"' in current_str else ""

            # 2. Extract params from the RAW string to preserve perfect index math
            params_str = current_str.split('"parameters"')[1] if '"parameters"' in current_str else ""

            if params_str:
                last_comma_indx = params_str.rfind(',')
                last_colon_indx = params_str.rfind(':')
                last_brace_idx = params_str.rfind('}')

                # 3. Calculate exactly where the AI cursor is
                is_inside_value = last_colon_indx > last_comma_indx and last_colon_indx > last_brace_idx

                # 4. Find the active key without the '$' trap
                active_key = ""
                if is_inside_value:
                    keys_found = re.findall(r'"([^"]+)"\s*:', params_str)
                    if keys_found:
                        active_key = keys_found[-1]

                expected_type = cache.param_types.get(func_name, {}).get(active_key, "Any")
                param_count = len(re.findall(r'"([^"]+)"\s*:', params_str))
                target_count = cache.func_params.get(func_name, 99)

                # --- 5. THE MASK ROUTER ---

                # RULE A: Strict Types override everything when inside a value
                if is_inside_value and expected_type == "number":
                    mask = cache.p4_numbers_only.copy() # Use .copy() to avoid mutating the global array!

                    if param_count == target_count:
                        for i, s in cache.clean_dict_items:
                            if ',' in s:
                                mask[i] = False

                # RULE B: String value state
                elif is_inside_value and params_str.count('"') % 2 != 0:
                    mask = cache.p4_mask.copy()

                # RULE C: Quota is MET. Force the close.
                elif param_count == target_count:
                    if current_str.strip().endswith('"') or current_str.strip().endswith(',') or current_str.strip().endswith('}'):
                        # Apply your custom logic to force the AI to close the JSON here
                        mask = cache.p4_no_comma.copy()
                    else:
                        mask = cache.p4_no_comma.copy()

                # RULE D: Quota NOT met. Waiting for next key.
                else:
                    mask = cache.p4_mask.copy()

        # Phase 1-3: Strict Spelling
        else:
            for i, s in cache.mini_dict:
                if any(rule.startswith(s) for rule in rules):
                    mask[i] = True

        logits[~mask] = -np.inf

        best_id = int(np.argmax(logits))
        current_str += cache.vocab_dict.get(best_id, "")
        input_ids.append(best_id)

        if current_str.endswith('"') and not bridge_injected and prefix in current_str:
            bridge = ',"parameters":{'
            current_str += bridge
            input_ids.extend(cache.model.encode(bridge).tolist()[0])
            bridge_injected = True
            continue

        else:
            print(f"\rGenerating: {current_str}", end="", flush=True)

    print()
    return current_str[:current_str.rfind('}') + 1] if '}' in current_str else current_str
