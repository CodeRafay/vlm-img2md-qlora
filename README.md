# VLM Markdown Generator (Qwen2-VL + QLoRA)

Streamlit app that converts document images into structured Markdown using a Qwen2-VL base model with a fine-tuned QLoRA adapter.

## Project structure

- app.py: Streamlit app
- qwen2-vl-nougat-qlora-final/: QLoRA adapter + tokenizer/processor config
- ai-ass05-22f-3327_22f-8803.ipynb: Training notebook

## Setup (local)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud notes

- The base model is loaded from Hugging Face at runtime.
- Keep only the adapter and tokenizer/processor files in this repo.
- The base model download is large and requires enough RAM and disk.

## Model files expected

The app expects this folder to exist in the repo:

- qwen2-vl-nougat-qlora-final/adapter_model.safetensors
- qwen2-vl-nougat-qlora-final/adapter_config.json
- qwen2-vl-nougat-qlora-final/tokenizer.json
- qwen2-vl-nougat-qlora-final/tokenizer_config.json
- qwen2-vl-nougat-qlora-final/processor_config.json
- qwen2-vl-nougat-qlora-final/chat_template.jinja

```

```
