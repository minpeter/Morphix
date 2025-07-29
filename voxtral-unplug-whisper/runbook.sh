hf download mistralai/Voxtral-Mini-3B-2507 --local-dir="voxtral-mini"

uv run 00-analyze.py ./voxtral-mini
uv run 01-conv-text-only-mistral.py ./voxtral-mini ./text_only_mistral
uv run 02-conv-mistral-to-hf.py ./text_only_mistral ./text_only_hf
uv run 03-conv-hf-tokenizer.py ./voxtral-mini ./text_only_hf


hf upload minpeter/Voxtral-Mini-3B-Text-2507 ./text_only_mistral --delete="*"
hf upload minpeter/Voxtral-Mini-3B-Text-2507-hf ./text_only_hf