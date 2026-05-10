# Call-Me-Maybe
This project bridges the gap between human language and machine-executable code by translating prompts into structured function calls with typed arguments. Built strictly with Python 3.10+ and Pydantic , it demonstrates how constrained decoding can ensure near-perfect reliability and strict JSON compliance, even on small 0.5B parameter models.


*This project has been created as part of the 42 curriculum by ayfadli.*

# Call Me Maybe 🤙
**An LLM Function Calling Tool powered by Constrained Decoding**

## Description
Small Language Models (like the Qwen3-0.6B model used in this project) are incredibly powerful, but they are notoriously bad at outputting structured, machine-readable data like JSON. If you ask a 0.6B model to output a function call, it will often hallucinate keys, forget commas, or add conversational text that crashes the system.

**Call Me Maybe** solves this problem without relying on massive parameter counts or "hope-based" prompt engineering. Instead, it implements a custom **Constrained Decoding Engine** (a Finite State Machine) that manipulates the model's token probabilities (logits) in real-time. By mathematically forcing the model to adhere to a strict JSON schema token-by-token, this project achieves 100% syntactically valid JSON output, turning a chaotic text generator into a highly reliable, autonomous function-calling agent.

---

## Algorithm Explanation: The Constrained Decoding "Bouncer"
The core of this project lives in `vocab_parser.py`. Instead of letting the LLM generate text freely, we intercept the probability distribution (logits) of all 150,000+ words in its vocabulary before it types a single character.

If a token violates our JSON structure, its probability is set to `-infinity`, making it mathematically impossible for the AI to select it. The algorithm operates in 4 distinct phases:

1. **Phase 1: The Prefix Cage:** The model is forced to type exactly `{"name":"`. All other tokens are blocked.
2. **Phase 2: The Dynamic Sandbox:** The engine reads the available function definitions (e.g., `fn_add_numbers`) and only allows tokens that perfectly spell one of the allowed function names.
3. **Phase 3: The Bridge:** Once a valid name is completed, the engine forces the model to type `","parameters":{`.
4. **Phase 4: The Parameter Sandbox:** The engine opens up to allow all printable characters (letters, numbers, punctuation) so the AI can reason about the user's prompt and extract the correct variables. A system prompt provides the schema blueprints.
5. **The Kill-Switch:** The moment the model forms the `}}` closing brackets, generation halts, and any trailing conversational garbage is truncated.

---

## Design Decisions
* **Separation of Concerns:** The project is split into two main logical components. `__main__.py` acts as the chassis (handling CLI arguments, Pydantic validation, and file I/O), while `vocab_parser.py` acts as the engine (handling the complex tensor math and logit manipulation).
* **Strict Token Matching:** To prevent the AI from bypassing constraints using multi-character tokens (e.g., jumping cages using the `":"` token), the validation logic uses strict `startswith()` prefix checking. A token can only be selected if it perfectly fits the *remaining* allowed string.
* **Schema Injection:** Even with constrained decoding, the AI needs to know *what* keys to generate. The loaded JSON schemas are dynamically injected into the system prompt to guide the AI's reasoning during Phase 4.

---

## Challenges Faced & Solutions
Building a custom decoder exposed several bizarre behaviors of LLM tokenizers:
* **The "Space Eater" Bug:** Initially, `token.strip()` was used to clean up formatting. However, Qwen tokens natively include spaces. Stripping them destroyed the spaces inside user parameters (e.g., `"HelloWorld"` instead of `"Hello World"`). The solution was to process tokens raw, allowing spaces to persist.
* **The Tokenizer Heist:** The model realized it could bypass the function-name validation by selecting tokens that contained both a quote and a colon (`":"`). This bypassed Phase 2 entirely. The solution was implementing an "Iron-Clad" cage that checks the exact string length of the remaining characters, closing the backdoor.
* **The Infinite Hallucination Loop:** If the AI tried to type a character that wasn't explicitly allowed in Phase 4 (like an apostrophe), it would get blocked, panic, and infinitely loop standard characters. This was solved by expanding the Phase 4 allowed list to `string.printable` and adding a hard `max_tokens` kill-switch.

---

## Performance Analysis
* **Accuracy:** The system achieves near 100% accuracy in outputting syntactically valid JSON. By dynamically loading the allowed function names, hallucinated function calls are completely eliminated.
* **Speed:** Because invalid tokens are aggressively zeroed out, the model spends less time exploring bad generation paths, resulting in fast, snappy function extractions.
* **Reliability:** Pydantic models (`FunctionDef` and `FunctionCallResult`) ensure that both the input schemas and the final outputs are strictly validated before being saved to disk.

---

## Testing Strategy
The implementation was iteratively tested against a variety of edge cases:
* Complex string manipulations involving punctuation and spaces.
* Number extraction (prompting the model strictly to ensure it outputs `42` instead of `"42"` to match the numeric schema).
* Multi-parameter functions requiring array-like or multi-key reasoning.

---

## Instructions (Usage)

### Prerequisites
* Python 3.10 or later
* `uv` package manager

### Installation
Clone the repository and install the dependencies (numpy, pydantic) into a virtual environment:
```bash
make install
# or
uv sync
