from llm_sdk import Small_LLM_Model

def main():
    #Wake the Robot:
    model = Small_LLM_Model()

    #A simple prompt string:
    prompt = "The capital of Morocco is"

    #Inward Translator(here we call Tokenization):
    input_IDs = model.encode(prompt).tolist()[0]
    print("\n\n--------\n")
    print(input_IDs)

    #Here we will get logits from input IDs:
    all_scores = model.get_logits_from_input_ids(input_ids=input_IDs)

    #Now we will get back from the list the winning ID
    max_score = max(all_scores)
    print(f"The index of high score: all_scores[{all_scores.index(max_score)}]={max_score}")
    with open("file.json", 'w') as f:
        print(all_scores, file=f)

    #Extract the decoded word using decode
    word = model.decode(all_scores.index(max_score))
    print(prompt + word)

if __name__ == "__main__":
    main()
