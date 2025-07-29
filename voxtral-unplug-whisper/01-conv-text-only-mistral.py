# /// script
# dependencies = [
#   "huggingface",
#   "safetensors",
#   "torch",
#   "numpy",
# ]
# ///

from safetensors.torch import safe_open, save_file
import json
import os
import re

STATE_DICT_MAPPING = {
    # CausalLM keys
    r"^output.weight":                               r"output.weight",
 
    # Model keys
    r"^norm.weight":                                 r"norm.weight",
    r"^mm_whisper_embeddings.tok_embeddings.weight": r"tok_embeddings.weight",

    # Layers keys
    r"^layers.(\d+).attention_norm.weight":          r"layers.\1.attention_norm.weight",
    r"^layers.(\d+).ffn_norm.weight":                r"layers.\1.ffn_norm.weight",

    # Attention keys
    r"^layers.(\d+).attention.w(q|k|v|o).weight":    r"layers.\1.attention.w\2.weight",


    # MLP keys
    r"^layers.(\d+).feed_forward.w1.weight":         r"layers.\1.feed_forward.w1.weight",
    r"^layers.(\d+).feed_forward.w2.weight":         r"layers.\1.feed_forward.w2.weight",
    r"^layers.(\d+).feed_forward.w3.weight":         r"layers.\1.feed_forward.w3.weight",
}

model_dir = "voxtral-mini"
consolidated_path = os.path.join(model_dir, "consolidated.safetensors")
tekken_path = os.path.join(model_dir, "tekken.json")
params_path = os.path.join(model_dir, "params.json")

text_only_tensors = {}
with safe_open(consolidated_path, framework="pt", device=0) as f:
    for k in f.keys():
        # Move tensors to text_only_tensors according to STATE_DICT_MAPPING.
        for pattern, replacement in STATE_DICT_MAPPING.items():
            if re.match(pattern, k):
                new_key = re.sub(pattern, replacement, k)
                text_only_tensors[new_key] = f.get_tensor(k)
                break
        else:
            # Skip keys that do not match any pattern in the mapping.
            continue

save_dir = "text_only"
consolidated_save_path = os.path.join(save_dir, "consolidated.safetensors")
tekken_save_path = os.path.join(save_dir, "tekken.json")
params_save_path = os.path.join(save_dir, "params.json")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_file(text_only_tensors, consolidated_save_path)

# Remove "audio" from tekken.json
with open(tekken_path, "r", encoding="utf-8") as f:
    tekken_data = json.load(f)
tekken_data.pop("audio", None)
with open(tekken_save_path, "w", encoding="utf-8") as f:
    json.dump(tekken_data, f, ensure_ascii=False, indent=2)

# Remove only "multimodal" from params.json
with open(params_path, "r", encoding="utf-8") as f:
    params_data = json.load(f)
params_data.pop("multimodal", None)
with open(params_save_path, "w", encoding="utf-8") as f:
    json.dump(params_data, f, ensure_ascii=False, indent=2)