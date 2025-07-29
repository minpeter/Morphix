# /// script
# dependencies = [
#   "transformers",
#   "safetensors",
#   "torch",
#   "numpy",
#   "mistral-common",
# ]
# ///

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Load Mistral tokenizer
mistral_tokenizer = MistralTokenizer.from_file("./voxtral-mini/tekken.json")

# Debug vocab mapping
print("=== Debugging Vocab Mapping ===")

# Get actual token mappings for a few test tokens
test_text = "Hello world!"
tokens = mistral_tokenizer.instruct_tokenizer.tokenizer.encode(test_text, bos=False, eos=False)
print(f"Text: {test_text}")
print(f"Tokens: {tokens}")

# Check vocab mappings
vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
print(f"Vocab size (no special): {len(vocab)}")

# Print first few tokens
print("\nFirst 10 vocab items:")
for i, (token, token_id) in enumerate(vocab.items()):
    if i >= 10:
        break
    print(f"  {token_id}: {repr(token)}")

# Check special tokens
print("\nSpecial tokens:")
for token in mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens:
    if hasattr(token, "value") and isinstance(token.value, dict):
        print(f"  {token.value['rank']}: {repr(token.value['token_str'])}")
    elif isinstance(token, dict):
        print(f"  {token['rank']}: {repr(token['token_str'])}")

# Try to find the issue with token "Hello"
print(f"\nLooking for 'Hello' token...")
hello_found = False
for token, token_id in vocab.items():
    if isinstance(token, bytes):
        try:
            token_str = token.decode('utf-8')
        except:
            token_str = str(token)
    else:
        token_str = str(token)
    
    if "Hello" in token_str or "hello" in token_str.lower():
        print(f"  Found: {token_id}: {repr(token)} -> {repr(token_str)}")
        hello_found = True
        if token_id == 22177:  # The expected ID from Mistral
            print(f"    ✅ Found exact match!")
        break

if not hello_found:
    print("  No 'Hello' token found in vocab")

# Check if there's an offset issue
print(f"\nChecking for potential offset issue...")
test_tokens = [22177, 4304, 1033]  # Expected Mistral tokens for "Hello world!"
for token_id in test_tokens:
    # Find the token with this ID
    found_token = None
    for token, tid in vocab.items():
        if tid == token_id:
            found_token = token
            break
    
    if found_token:
        if isinstance(found_token, bytes):
            try:
                token_str = found_token.decode('utf-8')
            except:
                token_str = str(found_token)
        else:
            token_str = str(found_token)
        print(f"  Token ID {token_id}: {repr(found_token)} -> {repr(token_str)}")
    else:
        print(f"  Token ID {token_id}: NOT FOUND in vocab")
