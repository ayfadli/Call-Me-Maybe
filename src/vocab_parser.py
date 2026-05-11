import string
import json


def get_allowed_chars(current_string, allowed_fn_names):
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


def run_bouncer(model, prompt_text, my_dict, allowed_fn_names, raw_functions):

    schema_hints = json.dumps(raw_functions)

    prompt = f"System: You are a strict API. Output ONLY valid JSON matching these schemas: {schema_hints}. CRITICAL: If a parameter is a 'number', output raw digits without quotes (e.g. 42, NOT \"42\").\nUser: {prompt_text}\nTool Call: "

    input_ids = model.encode(prompt).tolist()[0]

    current_string = ""
    max_tokens = 150
    token_count = 0

    while '}}' not in current_string.replace(" ", "").replace("\n", "") and token_count < max_tokens:

        allowed_rules = get_allowed_chars(current_string, allowed_fn_names)
        logits = model.get_logit_from_input_ids(input_ids)

        for token_id in range(len(logits)):
            token_str = my_dict.get(token_id)

            is_valid = False

            if not token_str:
                logits[token_id] = float('-inf')
                continue

            if (len(allowed_rules) > 10):
                if all(c in allowed_rules for c in token_str):
                    is_valid = True

            else:
                for rule in allowed_rules:
                    if rule.startswith(token_str):
                        is_valid = True
                        break

            if not is_valid:
                logits[token_id] = float('-inf')

        best_score = logits.index(max(logits))
        winning_string = my_dict.get(best_score)

        current_string += winning_string
        input_ids.append(best_score)

        token_count += 1
        print(f"\rGenerating: {current_string}", end="", flush=True)

    last_brace_idx = current_string.rfind('}')

    if last_brace_idx != -1:
        current_string = current_string[:last_brace_idx + 1]

    return current_string

