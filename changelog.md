* v2: less training because we overfit
* v3: fix 'max_seq_length', the prompts were truncated before getting to the observations and actions
* v4: special tokens
* v5: completion only training (not conversational)
* v6: remove special tokens, grpo on instruct model directly
* v7: no example in sys prompt. data format: "instructions: ... User: ... Controller: ..."
