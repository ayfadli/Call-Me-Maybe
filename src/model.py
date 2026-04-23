import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        print("Loading AI model, please wait...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model =  AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        print("Model loaded successfuly !")
        return model, tokenizer
    except Exception as e:
        print(f"An error occured: {e}", file=sys.stderr)
        sys.exit(1)

def generate_function_call(model, tokenizer, prompt: str,available_functions: list):
    tools = []
    for item in available_functions:
        formatted_tool = {
            "type": "function",
            "function": {
                "name": item.name,
                "description": item.description,
                "parameters": {
                    "type": "object",
                    "properties": {key: {"type": value.type} for key, value in item.parameters.items()}
                }
            }
        }
        tools.append(formatted_tool)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Your job is to extract data from user prompts and call the available functions."},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)

    input_lenght = inputs['input_ids'].shape[1]
    new_tokens = generated_ids[0][input_lenght:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response_text
