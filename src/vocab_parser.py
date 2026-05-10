import string
import json

def get_allowed_chars(current_string, allowed_fn_names):
    target_prefix = '{"name":"'
    bridge = '","parameters":{'

    # PHASE 1: The Prefix
    if len(current_string) < len(target_prefix):
        return [target_prefix[len(current_string):]]

    after_prefix = current_string[len(target_prefix):]

    # PHASE 2: Dynamic Name Sandbox (IRON-CLAD)
    if '"' not in after_prefix:
        allowed = [] # ABSOLUTELY NO QUOTES ALLOWED HERE!
        for name in allowed_fn_names:
            if name.startswith(after_prefix):
                # Give the Bouncer the exact remaining string PLUS the closing quote
                remaining_string = name[len(after_prefix):] + '"'
                allowed.append(remaining_string)
        return allowed

    # PHASE 3: The Bridge
    function_name = after_prefix.split('"')[0]
    perfect_string = target_prefix + function_name + bridge

    if len(current_string) < len(perfect_string):
        return [perfect_string[len(current_string):]]

    # PHASE 4: The Parameter Sandbox
    return list(string.printable)

def run_bouncer(model, prompt_text, my_dict, allowed_fn_names, raw_functions):
    schema_hints = json.dumps(raw_functions)
    # Strict prompt to enforce number types without quotes
    prompt = f"System: You are a strict API. Output ONLY valid JSON matching these schemas: {schema_hints}. CRITICAL: If a parameter is a 'number', output raw digits without quotes (e.g. 42, NOT \"42\").\nUser: {prompt_text}\nTool Call: "

    input_ids = model.encode(prompt).tolist()[0]
    current_string = ""

    max_tokens = 150
    token_count = 0

    # Kill-switch stops loop if "}}" is formed, even if the AI tries to sneak spaces in!
    while '}}' not in current_string.replace(" ", "").replace("\n", "") and token_count < max_tokens:
        allowed_rules = get_allowed_chars(current_string, allowed_fn_names)
        logits = model.get_logits_from_input_ids(input_ids)

        for token_id in range(len(logits)):
            token_str = my_dict.get(token_id)

            # Block pure empty tokens to prevent infinite looping
            if token_str is None or token_str == "":
                logits[token_id] = float('-inf')
                continue

            is_valid = False

            # PHASE 4: Free typing (but character checked)
            if len(allowed_rules) > 10:
                if all(c in allowed_rules for c in token_str):
                    is_valid = True
            # PHASE 1, 2, 3: Iron-clad exact match
            else:
                for rule in allowed_rules:
                    if rule.startswith(token_str):
                        is_valid = True
                        break

            if not is_valid:
                logits[token_id] = float('-inf')

        best_token_id = logits.index(max(logits))
        winning_str = my_dict[best_token_id]

        current_string += winning_str # KEEP THE SPACES!
        input_ids.append(best_token_id)
        token_count += 1

        print(f"\rGenerating: {current_string}", end="", flush=True)

    if token_count >= max_tokens:
        print("\n[WARNING] Max tokens reached! Forcing shutdown to prevent infinite loop.")

    # CLEANUP: Find the last '}' and chop off any conversational garbage the AI added
    last_brace_idx = current_string.rfind('}')
    if last_brace_idx != -1:
        current_string = current_string[:last_brace_idx + 1]

    print()
    return current_string
