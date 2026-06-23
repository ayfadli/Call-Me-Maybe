import string
import re
import numpy as np
from typing import Any, Dict
import json

def get_allowed_chars(current_str: str, allowed_names: list[str]) -> list[str]:
    #Phase 1
    prefix = '{"name":"'
    if len(current_str) < len(prefix):
        return [prefix[len(current_str):]]

    #Phase 2: The function name
    after_prefix = current_str[len(prefix):]
    if '"' not in after_prefix:
        return [name[len(after_prefix):] + '"' for name in allowed_names if name.startswith(after_prefix)]
    
    #Phase 3: The bridge
    func_name = after_prefix.split('"')[0]
    target = prefix + func_name + '","parameters":{'
    if len(current_str) < len(target):
        return [target[len(current_str):]]
    
    #Phase 4: The arguments
    return list(string.printable)

def generate_constrained_json(
    model: Any, prompt_text: str, vocab_dict: Dict,
    allowed_fn: list[str], raw_functions: list[Dict[str, Any]],
    p4_valid_ids: list[int], clean_dict_items: list[tuple[int, str]],
    func_params: Dict[str, int], visualize: bool) -> str:
    
    schema_hints = json.dumps(raw_functions)
    prompt = f"System: You are a strict API. Output only valid JSON matching these schemas: {schema_hints}. CRITICAL: If a parameter is a 'number', output float format without quotes. \nUser: {prompt_text}\nTool Call: "

    input_ids = model.encode(prompt).tolist()[0]
    vocab_size = np.array(model.get_logits_from_input_ids(input_ids)).shape[-1]

    target_phrases = allowed_fn + ['{"name":"', '","parameters":{', '}']
    mini_dict = [(i, s) for i, s in clean_dict_items if any(s in phrase for phrase in target_phrases)]

    p4_mask = np.zeros(vocab_size, dtype=bool)
    p4_mask[p4_valid_ids] = True

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
    just_rolled_back = False

    while '}}' not in current_str.replace(" ", "").replace("\n", "") and len(input_ids) < len(prompt) + max_tokens:
        
        rules = get_allowed_chars(current_str, allowed_fn)
        logits = np.array(model.get_logits_from_input_ids(input_ids))
        mask = np.zeros(vocab_size, dtype=bool)

        if visualize:
            # np.argsort sorts smallest to largest. [-3:] gets the last 3 (the highest). [::-1] reverses it to highest-first.
            top_3_ids = np.argsort(logits)[-3:][::-1]
            top_3_tokens = [model.decode([t]) for t in top_3_ids]
            scores = [round(logits[t], 2) for t in top_3_ids]

        if len(rules) > 10:
            # Phase 4: Quota Shield
            clean_curr = current_str.replace(" ", "")
            func_name = clean_curr.split('"name":"')[1].split('"')[0] if '"name":"' in current_str else ""
            params = clean_curr.split('"parameters"')[1] if '"parameters"' in current_str else ""

            # Enter if we are in the last parmeter
            if len(re.findall(r'"([^"]+)"\s*:', params)) == func_params.get(func_name, 99):
                if params.count('"') % 2 != 0:
                    mask = p4_mask # Inside string value
                elif current_str.strip().endswith('"'): # The last parameter the AI generated was a String.
                    for i, s in mini_dict:
                        if s.strip() == '}': # Force the AI to close the JSON 
                            mask[i] = True
                else:
                    mask = p4_no_comma
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
            just_rolled_back = True
            print("Dead end detected! Initiating rollback...")
            continue


        for bad_id in ids_blacklist:
            logits[bad_id] = -np.inf
        
        just_rolled_back = False

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
            top_3_masked_ids = np.argsort(logits)[-3:][::-1]
            top_3_masked_tokens = [model.decode([t]) for t in top_3_masked_ids]
            print(f"\n[AI Wanted]: {list(zip(top_3_tokens, scores))}")
            print(f"[We Forced]    : {top_3_masked_tokens[0]} (ID: {top_3_masked_ids[0]})")

        print(f"\rGenerating: {current_str}", end="", flush=True)

    print()
    return current_str[:current_str.rfind('}') + 1] if '}' in current_str else current_str
