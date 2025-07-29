# /// script
# dependencies = [
#   "huggingface",
#   "safetensors",
#   "torch",
#   "numpy",
# ]
# ///

import os
import json
import torch
from safetensors import safe_open
from collections import defaultdict

model_dir = "voxtral-mini"
index_path = os.path.join(model_dir, "model.safetensors.index.json")

try:
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
except FileNotFoundError:
    raise FileNotFoundError(
        f"필수 파일인 'model.safetensors.index.json'을 {model_dir}에서 찾을 수 없습니다."
    )

shards = defaultdict(list)
for tensor_name, shard_filename in weight_map.items():
    shards[shard_filename].append(tensor_name)

state_dict = {}

print("분할된 safetensor 파일 로드를 시작합니다...")
for shard_filename, tensor_names in shards.items():
    shard_path = os.path.join(model_dir, shard_filename)
    print(f"- {shard_filename} 로드 중...")
    
    with safe_open(shard_path, framework="pt", device=0) as f:
        for tensor_name in tensor_names:
            state_dict[tensor_name] = f.get_tensor(tensor_name)

print("\n✅ 로드 완료!")
print(f"총 {len(state_dict)}개의 텐서를 성공적으로 로드했습니다.")


consolidated_path = os.path.join(model_dir, "consolidated.safetensors")

consolidated_tensors = {}
with safe_open(consolidated_path, framework="pt", device=0) as f:
    for k in f.keys():
        consolidated_tensors[k] = f.get_tensor(k)

# r"^tok_embeddings.weight":  r"model.embed_tokens.weight",
# r"^output.weight":          r"lm_head.weight",

# 전부 로드되면 모델에서 language_model.lm_head.weight와 language_model.model.embed_tokens.weight를 비교
# tensor 크기, 값, 데이터 타입, 내부 내용
lm_head_weight = state_dict.get("language_model.lm_head.weight")
embed_tokens_weight = state_dict.get("language_model.model.embed_tokens.weight")

consolidated_tok_embeddings = consolidated_tensors.get("mm_whisper_embeddings.tok_embeddings.weight")
consolidated_output = consolidated_tensors.get("output.weight")

# 주어진 2개 텐서의 크기, 내용, 데이터 타입을 비교하는 함수 정의
def compare_tensors(tensor1, tensor2, name1, name2):
    if tensor1 is None or tensor2 is None:
        print(f"⚠️ {name1} 또는 {name2}가 None입니다.")
        return

    if tensor1.shape != tensor2.shape:
        print(f"❌ {name1}와 {name2}의 크기가 다릅니다: {tensor1.shape} vs {tensor2.shape}")
    else:
        print(f"✅ {name1}와 {name2}의 크기가 일치합니다: {tensor1.shape}")

    if tensor1.dtype != tensor2.dtype:
        print(f"❌ {name1}와 {name2}의 데이터 타입이 다릅니다: {tensor1.dtype} vs {tensor2.dtype}")
    else:
        print(f"✅ {name1}와 {name2}의 데이터 타입이 일치합니다: {tensor1.dtype}")

    # Ensure both tensors are on the same device before comparison
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    if not torch.equal(tensor1, tensor2):
        print(f"❌ {name1}와 {name2}의 내용이 다릅니다.")
    else:
        print(f"✅ {name1}와 {name2}의 내용이 일치합니다.")

compare_tensors(lm_head_weight, consolidated_output, "language_model.lm_head.weight", "output.weight")
compare_tensors(embed_tokens_weight, consolidated_tok_embeddings, "language_model.model.embed_tokens.weight", "mm_whisper_embeddings.tok_embeddings.weight")