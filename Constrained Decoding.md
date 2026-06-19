# Constrained Decoding Engine: The Finite State Machine

## Overview
The `run_bouncer` function is the core mathematical engine of the Call-Me-Maybe project. Because small Language Models (like Qwen 0.6B) are prone to hallucination and often output malformed JSON, we cannot rely on "prompt engineering" alone.

Instead, this function acts as a strict **Finite State Machine**. It intercepts the AI's generation process token-by-token, applying a mathematical mask (a "Bouncer") to the model's raw probability scores (`logits`). It forces the AI to output 100% valid, parseable JSON that strictly adheres to the provided schemas, while maintaining a sub-5-minute execution time.

---

## Core Architecture: The Mathematical Straitjacket

Text generation is fundamentally a loop. At each step, the model looks at the context and assigns a score (`logit`) to every single token in its 150,000-word vocabulary.

The `run_bouncer` intercepts this process using the following logic:

1. **Deny by Default:** We create a mock memory array (`masked_logits`) filled entirely with negative infinity (`-np.inf`). This mathematically locks down the entire vocabulary.
2. **Selective Permissions:** We calculate exactly which tokens are structurally legal at the current exact moment in the JSON string.
3. **The Score Copy:** We copy the AI's *real* scores for only the legal tokens into the locked-down array.
4. **Forced Selection:** We use `np.argmax()` to select the highest-scoring token. Because illegal tokens are trapped at negative infinity, the AI is mathematically forced to pick the best *legal* token.

---

## The 4 Phases of JSON Generation

To determine which characters are legal at any given microsecond, the engine evaluates the `current_string` buffer and categorizes the generation into four strict phases via the `get_allowed_chars` helper function.

### Phase 1: The Initialization (Target Prefix)
When the loop begins, the output buffer is empty. The AI might want to output conversational text like *"Sure, I can help with that."* * **The Rule:** The state machine strictly blocks all tokens except those that build the exact string `{"name":"`.
* **The Result:** The AI is instantly forced into the JSON dimension.

### Phase 2: Function Name Selection
Once the prefix is built, the AI must choose which tool to call.
* **The Rule:** The engine reads the `allowed_fn_names` list derived from the `functions_definition.json` schema. It filters the vocabulary to **only** allow tokens that perfectly match the spelling of the available functions (e.g., `fn_add_numbers`).
* **The Result:** It is physically impossible for the AI to hallucinate a non-existent function name.

### Phase 3: The Bridge
Once the function name is complete (detected by the closing quote `"`), the state machine must transition the JSON object into the parameters section.
* **The Rule:** The engine forces the generation of the exact bridge string: `","parameters":{`.
* **The Result:** The JSON structure remains flawlessly compliant, preventing missing commas or broken brackets.

### Phase 4: Parameter Generation (The Wild West)
Once inside the `parameters` dictionary, the AI must actually answer the user's prompt. It needs freedom to generate numbers, strings, and punctuation.
* **The Rule:** The state machine relaxes the strict string-matching and switches to allowing anything within `string.printable` (standard English letters, numbers, and basic symbols).

#### The `phase_4_valid_ids` Speed Hack
Evaluating 150,000 tokens against 100 printable characters takes massive CPU time if done on every loop iteration. To bypass this and beat the 5-minute performance requirement, we use an $O(1)$ memory swap:

```python
if (len(allowed_rules) > 10):
    # We are in Phase 4. Bypass calculations!
    valid_ids = phase_4_valid_ids
