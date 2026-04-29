from llm_sdk import Small_LLM_Model
import string

def get_allowed_chars(current_string):

    target_prefix = '{"name": "'

    #Check if the string is shorter than target_prefix that means we are in the 1st phase
    if len(current_string) < len(target_prefix):
        return [target_prefix[len(current_string):]]

    #Return a list of allowed characters: all lowercase letters, the underscore, and the quote mark "
    # (so the AI can eventually close the word).
    elif (current_string.count('"') == 3):
        allowed = list(string.ascii_lowercase) + ['_', '"']
        return allowed

    return ["BLOCK"]


def main():
    current_string = ""

    while(not current_string.endswith('}')):
        current_string = get_allowed_chars(current_string)
        print(current_string)
        break

    # print(current_string.count('"'))
    # allowed = get_allowed_chars(current_string)
    # print(allowed)

if __name__ == "__main__":
    main()
