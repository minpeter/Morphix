hf download mistralai/Voxtral-Mini-3B-2507 --local-dir="voxtral-mini"

uv run 01-conv-text-only-mistral.py
uv run 02-conv-mistral-to-hf.py ./text_only ./text_only

hf upload minpeter/Voxtral-Mini-3B-Text-2507 ./text_only