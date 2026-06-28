"""
Phase 1: Understanding Tokens and Roles
Run this with just: pip install tiktoken
No API key needed for this script.
"""
 
import tiktoken  # OpenAI's open-source tokenizer library
 
# 1. TOKENS: the model reads chunks, not words
encoder = tiktoken.get_encoding("cl100k_base")  # tokenizer used by the GPT-4 family
 
sample_text = "Pakistan tech scene is growing fast in 2026!"
tokens = encoder.encode(sample_text)
print(f"Original text: {sample_text}")
print(f"Token count:   {len(tokens)}")
 
# Decode each token ID back to text so you can literally SEE the splits
print("\nHow the tokenizer split this sentence:")
for token_id in tokens:
    piece = encoder.decode([token_id])
    print(f"  id={token_id:<8} -> {piece!r}")
 
# 2. CONTEXT WINDOW: a hard budget, not a soft suggestion
# Real limits vary a lot by provider and model (commonly 100K-1M+ tokens in 2026),
# but the PRINCIPLE is the same everywhere: system + history + reply must all fit.
CONTEXT_WINDOW_LIMIT = 200_000  # example only -- always check your specific model's docs
print(f"\nThis prompt uses {len(tokens)} of your ~{CONTEXT_WINDOW_LIMIT}-token budget.")
 
# 3. ROLE SYSTEM: a conversation is just a list of labeled dictionaries
conversation = [
    {"role": "system", "content": "You are a concise tutor for first-year CS students."},
    {"role": "user", "content": "What is a token, in one sentence?"},
]
print("\nA 'conversation' is nothing more than this list:")
for msg in conversation:
    print(f"  [{msg['role'].upper()}] {msg['content']}")
 
# 4. TEMPERATURE: you don't need an API call to understand the idea.
# temperature=0    -> always the single most likely next token (deterministic)
# temperature=0.7  -> balanced sampling from likely tokens (typical default)
# temperature=1.5+ -> samples from a much wider pool (more surprising, less reliable)
print("\nTemperature changes HOW the model picks the next token, not what the tokens are.")
print("You'll set this for real starting in Phase 2.")