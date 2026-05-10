from llm_sdk import Small_LLM_Model
import string
import json
import sys

def get_allowed_chars(current_string):

    target_prefix = '{"name":"'
    target_suffix = '","parameters":{'

    #Check if the string is shorter than target_prefix that means we are in the 1st phase
    if len(current_string) < len(target_prefix):
        return [target_prefix[len(current_string):]]

    after_prefix = current_string[len(target_prefix):]
    # Return a list of allowed characters: all lowercase letters, the underscore, and the quote mark "
    # (so the AI can eventually close the word).
    if '"' not in after_prefix:
        allowed = list(string.ascii_lowercase) + ['_', '"']
        return allowed

    function_name = after_prefix.split('"')[0]
    perfect_string = target_prefix + function_name + target_suffix

    if len(current_string) < len(perfect_string):
        return [perfect_string[len(current_string):]]

    json_chars = list(string.ascii_letters) + list(string.digits) + ['{', '}', '"', ':', ',', '_', '-', ' ']

    return json_chars


def main():
    current_string = ""

    #Boot the AI
    model = Small_LLM_Model()

    vocab_file = model.get_path_to_vocab_file()
    try:
        with open(vocab_file, 'r') as v:
            my_dict = json.load(v)
    except json.JSONDecodeError as e:
        print(f"Error: '{v}' contains invalid JSON (corrupted or missing a comma).\nDetails: {e}")
    except Exception as e:
        print(f"An enxepected error occured while reading: '{v}': {e}", file=sys.stderr)

    my_dict = {v: k.replace('Ġ', ' ') for k, v in my_dict.items()}


    prompt = """System: You have one tool.
    Name: "weather"
    Parameters: {"city": "string"}

    User: What is the weather like in Tokyo today?
    Assistant: I will use the weather tool to find out.
    Tool Call: """
    input_ids = model.encode(prompt).tolist()[0]

    while '}}' not in current_string:

        allowed_rules = get_allowed_chars(current_string)

        logits = model.get_logits_from_input_ids(input_ids)

        for token_id in range(len(logits)):

            token_str = my_dict.get(token_id)

            if token_str is None:
                logits[token_id] = float('-inf')
                continue

            clean_token = token_str.strip()
            if clean_token == "":
                    logits[token_id] = float('-inf')
                    continue

            is_valid = False

            if "BLOCK" in allowed_rules:
                logits[token_id] = float('-inf')
                continue

            for rule in allowed_rules:
                if rule.startswith(clean_token) or clean_token.startswith(rule):
                    is_valid = True
                    break

            if not is_valid:
                logits[token_id] = float('-inf')

        # if max(logits) == float('-inf'):
        #     print("\n\n[FATAL ERROR] The Bouncer blocked every single token! The cage is broken.")
        #     break

        best_token_id = logits.index(max(logits))
        winning_str = my_dict[best_token_id]

        current_string += winning_str.strip()
        input_ids.append(best_token_id)

        print(f"\rBuilding: {current_string}", end="", flush=True)

    current_string = current_string.split('}}')[0] + '}}'
    print()
    print(current_string)

    # print(current_string.count('"'))
    # allowed = get_allowed_chars(current_string)
    # print(allowed)

if __name__ == "__main__":
    main()
