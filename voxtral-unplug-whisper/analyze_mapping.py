# /// script
# dependencies = [
#   "transformers",
#   "mistral-common",
# ]
# ///

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

# Load Mistral tokenizer
mistral_tokenizer = MistralTokenizer.from_file("./voxtral-mini/tekken.json")
tekkenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

print("=== Analyzing the real token mapping ===")

# Test tokens
test_tokens = [22177, 4304, 1033]  # "Hello", " world", "!"

# Let's build the reverse mapping by encoding/decoding
print("Building reverse mapping...")
token_to_string = {}

# Try to map all tokens by decoding them individually
print("Sampling some tokens to understand the pattern:")
for i in range(0, 1000, 100):  # Special tokens range
    try:
        decoded = tekkenizer.decode([i])
        print(f"  Special token {i}: {repr(decoded)}")
    except:
        pass

for i in range(1000, 2000, 200):  # Regular tokens range
    try:
        decoded = tekkenizer.decode([i])
        print(f"  Regular token {i}: {repr(decoded)}")
    except:
        pass

# Now let's check our specific tokens
print(f"\nOur test tokens:")
for token_id in test_tokens:
    decoded = tekkenizer.decode([token_id])
    print(f"  Token {token_id}: {repr(decoded)}")

# Let's try to understand the mapping by looking at the vocab structure
print(f"\nVocab analysis:")
raw_vocab = tekkenizer._tekken_token2id_nospecial
byte_encoder = bytes_to_unicode()

# Find what maps to our token IDs in the raw vocab
print("Checking raw vocab for our token IDs:")
for token_id in test_tokens:
    found = False
    for token_bytes, tid in raw_vocab.items():
        if tid == token_id:
            print(f"  Token ID {token_id} -> raw: {repr(token_bytes)}")
            if isinstance(token_bytes, bytes):
                try:
                    decoded_bytes = token_bytes.decode('utf-8')
                    print(f"    UTF-8 decode: {repr(decoded_bytes)}")
                except:
                    pass
                # Try byte-to-unicode mapping
                try:
                    bpe_string = "".join([byte_encoder[b] for b in token_bytes])
                    print(f"    BPE string: {repr(bpe_string)}")
                except:
                    pass
            found = True
            break
    if not found:
        print(f"  Token ID {token_id}: NOT FOUND in raw vocab")

# Let's see if the real mapping might be offset
print(f"\nTrying offset theory:")
for token_id in test_tokens:
    # Try subtracting special token count
    adjusted_id = token_id - 1000
    if 0 <= adjusted_id < len(raw_vocab):
        # Find token with this adjusted ID
        for token_bytes, tid in raw_vocab.items():
            if tid == adjusted_id:
                print(f"  Token {token_id} -> adjusted {adjusted_id} -> {repr(token_bytes)}")
                decoded = tekkenizer.decode([token_id])
                print(f"    Actual decode: {repr(decoded)}")
                break

# Check the highest ID
max_id = max(raw_vocab.values())
print(f"\nMax regular token ID: {max_id}")
print(f"Special tokens: 0-999")
print(f"Regular tokens seem to be: 1000-{max_id + 1000}")

# Let's test this theory by creating a corrected mapping
print(f"\nTesting corrected mapping:")
corrected_vocab = {}
special_count = 1000

# Add special tokens
for token in tekkenizer._all_special_tokens:
    if hasattr(token, 'value') and isinstance(token.value, dict):
        token_str = token.value['token_str']
        token_id = token.value['rank']
        corrected_vocab[token_str] = token_id
        if token_id < 10:  # Show first 10
            print(f"  Special {token_id}: {repr(token_str)}")

# Add regular tokens with offset
for token_bytes, original_id in raw_vocab.items():
    corrected_id = original_id + special_count
    if isinstance(token_bytes, bytes):
        bpe_string = "".join([byte_encoder[b] for b in token_bytes])
        corrected_vocab[bpe_string] = corrected_id
        if corrected_id in test_tokens:
            print(f"  Regular {corrected_id}: {repr(bpe_string)} (from {repr(token_bytes)})")

print(f"\nCorrected vocab size: {len(corrected_vocab)}")
