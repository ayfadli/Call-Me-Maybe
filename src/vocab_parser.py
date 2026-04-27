from llm_sdk import Small_LLM_Model
import json
import sys

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
    print(my_dict[9625])

if __name__ == "__main__":
    main()
