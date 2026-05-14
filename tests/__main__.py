import unittest
import math
import re
import json

def fn_add_numbers(a, b):
    """Add two numbers together and return their sum."""
    return a + b

def fn_greet(name):
    return f"Hello, {name}."

def fn_reverse_string(s):
    return s[::-1]

def fn_get_square_root(a):
    if a < 0:
        return "The number connot be negative."
    return math.sqrt(a)

def fn_substitute_string_with_regex(source_string, regex, replacement):
    new_str = re.sub(regex, replacement, source_string)
    return new_str


def main():
    functions = {
        "fn_add_numbers": fn_add_numbers,
        "fn_greet": fn_greet,
        "fn_reverse_string": fn_reverse_string,
        "fn_get_square_root": fn_get_square_root,
        "fn_substitute_string_with_regex": fn_substitute_string_with_regex
    }

    with open("/goinfre/ayfadli/Call-Me-Maybe/data/output/function_calling_results.json", 'r') as output_file:
        data = json.load(output_file)
        for test in data:
            print(test['prompt'])

            fun_name = test['name']
            args = test['parameters']

            result = functions[fun_name](**args)

            print(result)
            print()
        # print(data[0])

if __name__ == "__main__":
    main()
