# /// script
# dependencies = [
#   "transformers",
#   "safetensors",
#   "torch",
#   "numpy",
#   "mistral-common",
# ]
# ///

import argparse
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import LlamaTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode


class MistralConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, vocab: dict):
        bpe_ranks = vocab
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            if isinstance(b, bytes):
                return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])
            else:
                return b  # Already a string

        merges = []
        vocab = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in self.additional_special_tokens:
                if isinstance(token, bytes):
                    vocab[token_bytes_to_string(token)] = rank
                else:
                    vocab[token] = rank
                    
                if isinstance(token, bytes) and len(token) == 1:
                    continue
                elif isinstance(token, str) and len(token) == 1:
                    continue
                    
                local = []
                if isinstance(token, bytes):
                    for index in range(1, len(token)):
                        piece_l, piece_r = token[:index], token[index:]
                        if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                            local.append((piece_l, piece_r, rank))
                    local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
                    merges.extend([(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in local])
                # Skip merge generation for string tokens (they should already be properly formatted)
            else:
                vocab[token] = rank
                
        merges = sorted(merges, key=lambda val: (vocab.get(val[0], 0), vocab.get(val[1], 0)), reverse=False)
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


def convert_tekken_tokenizer(tokenizer_file: str):
    """Convert a "tekken" tokenizer to a fast Tokenizer."""
    # Tekken format -- need to use the Converter

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    # Load directly using their lib
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)

    # Extract vocab and special tokens using the correct mapping
    raw_vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
    all_special = [
        token.value if hasattr(token, "value") else token
        for token in mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens
    ]

    # Extract special token strings for the converter
    special_token_strings = []
    special_tokens_dict = {}
    
    for token in all_special:
        if isinstance(token, dict) and "token_str" in token:
            token_str = token["token_str"]
            token_id = token["rank"]
            special_token_strings.append(token_str)
            special_tokens_dict[token_str] = token_id
        else:
            token_str = token if isinstance(token, str) else str(token)
            special_token_strings.append(token_str)

    # Create the full vocab with correct mapping
    # Mistral uses: special tokens 0-999, regular tokens 1000+
    full_vocab = {}
    byte_encoder = bytes_to_unicode()
    
    # First, add special tokens with their correct IDs (0-999)
    for token_str, token_id in special_tokens_dict.items():
        full_vocab[token_str] = token_id
    
    # Then, add regular tokens with proper byte encoding
    # The raw vocab has IDs 0-130071, but actual usage is 1000-131071
    for token_bytes, raw_id in raw_vocab.items():
        actual_id = raw_id + 1000  # Add the special token offset
        if isinstance(token_bytes, bytes):
            # Convert bytes to proper string representation for BPE
            token_str = "".join([byte_encoder[b] for b in token_bytes])
            full_vocab[token_str] = actual_id

    # Convert
    tokenizer = LlamaTokenizerFast(
        tokenizer_object=MistralConverter(vocab=full_vocab, additional_special_tokens=special_token_strings).converted(),
        legacy=False,  # Use new behavior
    )

    # Post-process
    tokenizer.add_special_tokens({"additional_special_tokens": special_token_strings})
    
    # Set special tokens properly
    if "<unk>" in special_token_strings:
        tokenizer.unk_token = "<unk>"
    if "<s>" in special_token_strings:
        tokenizer.bos_token = "<s>"
    if "</s>" in special_token_strings:
        tokenizer.eos_token = "</s>"
    if "<pad>" in special_token_strings:
        tokenizer.pad_token = "<pad>"
    
    # Add Mistral chat template - exact match to Mistral format with system prompt support
    chat_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token }}{% if system_message != '' %}{{ '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST]' + message['content'] + '[/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"""
    tokenizer.chat_template = chat_template

    return tokenizer


def test_tokenizer_consistency(hf_tokenizer, mistral_tokenizer):
    """Test if HuggingFace tokenizer produces same results as Mistral tokenizer."""
    
    from mistral_common.protocol.instruct.messages import (
        AssistantMessage,
        SystemMessage,
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    
    # Test cases
    test_cases = [
        # Simple text
        "Hello world!",
        "What is the capital of France?",
        "The capital of France is Paris.",
        
        # Text with special characters
        "café résumé naïve",
        "Hello 世界! 🌍",
    ]
    
    print("\n=== Testing Tokenizer Consistency ===")
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Text: {repr(test_case)}")
        
        # HF tokenizer
        hf_tokens = hf_tokenizer.encode(test_case, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_tokens)
        
        # Mistral tokenizer - use the instruct tokenizer directly
        mistral_tokens = mistral_tokenizer.instruct_tokenizer.tokenizer.encode(test_case, bos=False, eos=False)
        mistral_decoded = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(mistral_tokens)
        
        print(f"HF tokens:      {hf_tokens}")
        print(f"Mistral tokens: {mistral_tokens}")
        print(f"HF decoded:     {repr(hf_decoded)}")
        print(f"Mistral decoded: {repr(mistral_decoded)}")
        
        # Check consistency
        tokens_match = hf_tokens == mistral_tokens
        decoded_match = hf_decoded == mistral_decoded
        
        print(f"Tokens match: {tokens_match}")
        print(f"Decoded match: {decoded_match}")
        
        if not tokens_match or not decoded_match:
            print("❌ MISMATCH DETECTED!")
        else:
            print("✅ Match!")
    
    # Test chat messages separately
    print(f"\nChat Template Test:")
    chat_messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And the capital of Spain?"},
    ]
    
    print(f"Chat messages: {chat_messages}")
    
    # Convert to Mistral format
    messages = []
    for msg in chat_messages:
        if msg["role"] == "user":
            messages.append(UserMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AssistantMessage(content=msg["content"]))
    
    # Mistral tokenizer with chat format
    request = ChatCompletionRequest(messages=messages)
    mistral_tokenized = mistral_tokenizer.encode_chat_completion(request)
    mistral_tokens = mistral_tokenized.tokens
    mistral_text = mistral_tokenized.text
    
    # HF tokenizer with chat template
    hf_text = hf_tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    hf_tokens = hf_tokenizer.encode(hf_text, add_special_tokens=False)
    
    print(f"Mistral tokens: {mistral_tokens}")
    print(f"HF tokens:      {hf_tokens}")
    print(f"Mistral text:   {repr(mistral_text)}")
    print(f"HF text:        {repr(hf_text)}")
    
    # Check consistency
    tokens_match = hf_tokens == mistral_tokens
    text_match = hf_text.strip() == mistral_text.strip()
    
    print(f"Tokens match: {tokens_match}")
    print(f"Text match: {text_match}")
    
    if not tokens_match or not text_match:
        print("❌ CHAT TEMPLATE MISMATCH DETECTED!")
        # Let's analyze the differences
        if not text_match:
            print(f"Expected (Mistral): {repr(mistral_text)}")
            print(f"Got (HF):           {repr(hf_text)}")
    else:
        print("✅ Chat Template Match!")
    
    # Test with system message
    print(f"\nChat Template Test with System Message:")
    system_chat_messages = [
        {"role": "system", "content": "You are a helpful geography assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And the capital of Spain?"},
    ]
    
    print(f"System chat messages: {system_chat_messages}")
    
    # Convert to Mistral format with system message
    from mistral_common.protocol.instruct.messages import SystemMessage
    system_messages = []
    for msg in system_chat_messages:
        if msg["role"] == "system":
            system_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            system_messages.append(UserMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            system_messages.append(AssistantMessage(content=msg["content"]))
    
    # Mistral tokenizer with system message
    system_request = ChatCompletionRequest(messages=system_messages)
    mistral_system_tokenized = mistral_tokenizer.encode_chat_completion(system_request)
    mistral_system_tokens = mistral_system_tokenized.tokens
    mistral_system_text = mistral_system_tokenized.text
    
    # HF tokenizer with system message
    hf_system_text = hf_tokenizer.apply_chat_template(system_chat_messages, tokenize=False, add_generation_prompt=True)
    hf_system_tokens = hf_tokenizer.encode(hf_system_text, add_special_tokens=False)
    
    print(f"Mistral system tokens: {mistral_system_tokens}")
    print(f"HF system tokens:      {hf_system_tokens}")
    print(f"Mistral system text:   {repr(mistral_system_text)}")
    print(f"HF system text:        {repr(hf_system_text)}")
    
    # Check system message consistency
    system_tokens_match = hf_system_tokens == mistral_system_tokens
    system_text_match = hf_system_text.strip() == mistral_system_text.strip()
    
    print(f"System tokens match: {system_tokens_match}")
    print(f"System text match: {system_text_match}")
    
    if not system_tokens_match or not system_text_match:
        print("❌ SYSTEM CHAT TEMPLATE MISMATCH DETECTED!")
        if not system_text_match:
            print(f"Expected (Mistral): {repr(mistral_system_text)}")
            print(f"Got (HF):           {repr(hf_system_text)}")
    else:
        print("✅ System Chat Template Match!")
    
    print("\n=== Test Complete ===\n")

def convert_tokenizer_and_test(input_dir: str, output_dir: str):
    """Convert Tekken tokenizer to HuggingFace format and run tests."""
    import os
    
    tekken_path = os.path.join(input_dir, "tekken.json")
    
    # Convert tokenizer
    print("Converting Tekken tokenizer to HuggingFace format...")
    hf_tokenizer = convert_tekken_tokenizer(tekken_path)
    print(f"✅ Conversion complete! Vocab size: {len(hf_tokenizer)}")
    
    # Save the tokenizer
    hf_tokenizer.save_pretrained(output_dir)
    print(f"✅ Tokenizer saved to '{output_dir}' directory")

    # Load Mistral tokenizer for comparison
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    mistral_tokenizer = MistralTokenizer.from_file(tekken_path)
    
    # Run consistency tests
    test_tokenizer_consistency(hf_tokenizer, mistral_tokenizer)
    
    # Example usage with chat template
    print("=== Testing Chat Template ===")
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And the capital of Spain?"},
    ]
    
    # Test chat template
    formatted_chat = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted chat: {repr(formatted_chat)}")
    
    # Test tokenization of chat
    chat_tokens = hf_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    print(f"Chat tokens: {chat_tokens}")
    
    # Compare with original Mistral approach
    print("\n=== Original Mistral Tokenization ===")
    from mistral_common.protocol.instruct.messages import (
        AssistantMessage,
        UserMessage,
    )
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    mistral_messages = [
        UserMessage(content="What is the capital of France?"),
        AssistantMessage(content="The capital of France is Paris."),
        UserMessage(content="And the capital of Spain?"),
    ]

    request = ChatCompletionRequest(messages=mistral_messages)
    tokenized = mistral_tokenizer.encode_chat_completion(request)

    print(f"Mistral tokens: {tokenized.tokens}")
    print(f"Mistral text: {repr(tokenized.text)}")
    
    # Final comparison
    print(f"\nFinal comparison:")
    print(f"HF chat tokens:      {chat_tokens}")
    print(f"Mistral chat tokens: {tokenized.tokens}")
    print(f"Match: {chat_tokens == tokenized.tokens}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "output_dir",
        help="Location to write HF model and tokenizer",
    )
    
    args = parser.parse_args()
    convert_tokenizer_and_test(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()