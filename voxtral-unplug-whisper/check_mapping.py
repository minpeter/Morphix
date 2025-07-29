# /// script
# dependencies = [
#   "transformers",
#   "mistral-common",
# ]
# ///

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Load Mistral tokenizer
mistral_tokenizer = MistralTokenizer.from_file("./voxtral-mini/tekken.json")

# Get the complete token-to-id mapping that Mistral actually uses
print("=== Checking Mistral's actual token mapping ===")

# Test with "Hello world!"
test_text = "Hello world!"
tokens = mistral_tokenizer.instruct_tokenizer.tokenizer.encode(test_text, bos=False, eos=False)
print(f"Text: {test_text}")
print(f"Tokens: {tokens}")

# Get the actual token mapping including special tokens
print("Available attributes:")
tekkenizer = mistral_tokenizer.instruct_tokenizer.tokenizer
for attr in dir(tekkenizer):
    if 'token' in attr.lower() or 'vocab' in attr.lower():
        print(f"  {attr}")

# Let's check if we can get the full mapping differently
print(f"\nNo-special vocab size: {len(tekkenizer._tekken_token2id_nospecial)}")

# Let's try to decode the token IDs directly
print(f"\nDecoding tokens directly:")
for token_id in tokens:
    try:
        decoded = tekkenizer.decode([token_id])
        print(f"  Token ID {token_id}: {repr(decoded)}")
    except Exception as e:
        print(f"  Token ID {token_id}: Error - {e}")

# Let's check the special tokens and their IDs
print(f"\nSpecial tokens:")
for i, special_token in enumerate(tekkenizer._all_special_tokens[:10]):  # First 10
    if hasattr(special_token, 'value') and isinstance(special_token.value, dict):
        print(f"  {special_token.value['rank']}: {special_token.value['token_str']}")
    elif isinstance(special_token, dict):
        print(f"  {special_token['rank']}: {special_token['token_str']}")

# Check if we can find token 22177 in the no-special vocab
found_22177 = False
for token, tid in tekkenizer._tekken_token2id_nospecial.items():
    if tid == 22177:
        print(f"\nFound token 22177 in no-special vocab: {repr(token)}")
        found_22177 = True
        break

if not found_22177:
    print(f"\nToken 22177 NOT found in no-special vocab")
    # Check around that range
    print("Tokens around 22177:")
    for token, tid in tekkenizer._tekken_token2id_nospecial.items():
        if 22170 <= tid <= 22180:
            print(f"  {tid}: {repr(token)}")

# Let's check what the maximum ID in no-special vocab is
max_id = max(tekkenizer._tekken_token2id_nospecial.values())
min_id = min(tekkenizer._tekken_token2id_nospecial.values())
print(f"\nNo-special vocab ID range: {min_id} to {max_id}")
