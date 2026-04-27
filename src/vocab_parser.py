from llm_sdk import Small_LLM_Model
import json
import sys
import string

def get_allowed_chars(current_string):
    # The static part of the JSON that never changes
    static_prefix = '{"name": "'

    if  static_prefix.startswith(current_string) and current_string != static_prefix:
            next_required_char = static_prefix[len(current_string)]
            return next_required_char

    elif current_string.startswith(static_prefix):
        allowed = list(string.ascii_lowercase) + ['_', '"']
        return allowed

    return []


def main():
    model = Small_LLM_Model()

    vocab_file = model.get_path_to_vocab_file()
    vocab_data = {}

    try:
        with open(vocab_file, 'r') as v:
            vocab_data = json.load(v)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON data (corrupted or missing a comma).")
        print(f"Details: {e}", file=sys.stderr)
    except (FileNotFoundError, PermissionError, Exception) as e:
        print(f"An error occured while opening this file '{vocab_file}'")
        print(f"Details: {e}")

   #Reverse (map) the key and value pairs in a dict:
    my_dict = {v: k.replace('Ġ', ' ') for k, v in vocab_data.items()}

    prompt = "The capital of France is"

    input_ids = model.encode(prompt)

    # --- THE FIX: Sanitize the IDs into a flat Python list ---
    # 1. If encode() returned a PyTorch tensor, convert it to a Python list
    if hasattr(input_ids, 'tolist'):
        input_ids = input_ids.tolist()

    # 2. If encode() returned a 2D list (e.g., [[785, 6722]]), flatten it to 1D
    if len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    generated_json = ""

    logits = model.get_logits_from_input_ids(input_ids)

    print("--- Forcing JSON Output ---")
    while not generated_json.endswith('}'):

        logits = model.get_logits_from_input_ids(input_ids)
        allowed_chars = get_allowed_chars(generated_json)

        for token_id in range(len(logits)):

            token_str = my_dict.get(token_id)

            if token_str is None:
                logits[token_id] = float('-inf')
                continue

            if len(allowed_chars) > 0:

                is_valid = False

                clean_token = token_str.lstrip()

                for allowed in allowed_chars:
                    if clean_token.startswith(allowed):
                        is_valid = True
                        break

                if not is_valid:
                    logits[token_id] = float('-inf')

        best_token_id = logits.index(max(logits))
        winning_str = my_dict[best_token_id]

        if generated_json == "":
            winning_str = winning_str.lstrip()

        generated_json += winning_str
        input_ids.append(best_token_id)

        print(f"\rCurrent Output: {generated_json}", end="", flush=True)

        if len(generated_json) > 50:
            print("\n[Safety Break] String got too long without closing!")
            break


    print(f"\n\nFinal JSON: {generated_json}")


if __name__ == "__main__":
    main()
