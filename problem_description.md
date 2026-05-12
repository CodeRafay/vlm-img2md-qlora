Question: Vision Language Models (VLM) using QLoRA Fine-
Tuning for Document-to-Markdown Generation

Objective: The objective of this assignment is to fine-tune a Vision Language Model (VLM) using QLoRA
for the task of converting document images into structured Markdown documentation.
You will:
• Understand multimodal learning (image + text)
• Perform parameter-efficient fine-tuning
• Adapt a pretrained VLM to document understanding
• Generate Markdown outputs from unseen document images
1. Environment Setup
• Platform: https://www.kaggle.com/
• Accelerator: GPU T4 x2 (Dual GPU)
• Framework: PyTorch + Hugging Face Transformers + PEFT
• Dataset:

➢ Nougat Training Dataset Example: https://www.kaggle.com/datasets/zphilip/nougat-
training-dataset-example

2. Model Architecture
➢ Vision Language Model
Use the pretrained Qwen2-VL-2B-Instruct.
Model page: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

3. Fine-Tuning Technique
➢ QLoRA (Quantized Low-Rank Adaptation)
Instead of full fine tuning:
o load the base model in 4-bit quantized form
o Freeze original pretrained weights
o Train only lightweight LoRA adapters
4. Implementation Tasks
Part 1: Dataset Exploration
1. Load the Nougat dataset
2. Inspect image – markdown pairs
3. Understand visual layout and document structure
4. Select a subset of samples or full dataset for training
Part 2: Data Preparation
Convert the dataset into ChatML format. Each sample should contain:
1. Image
2. Instruction prompt
3. Markdown target
Part 3: Dataset Subsetting
Recommended split:
o 80% training
o 20% validation
Or use can use your specified settings

Part 4: QLoRA Fine-Tuning
Fine-tune the model using:
• 4-bit quantization
• LoRA adapters
• 2–5 epochs

Required Training Techniques (to fit Kaggle T4×2)

• batch size: 1–2
• gradient accumulation: 4–8
• learning rate: 1e-4 to 2e-4
• LoRA rank: 8 or 16
• image resolution: 512–768 px
• or your recommended settings
Part 5: Markdown Generation (CORE TASK)
For each validation image:
1. Run the fine-tuned model

2. Generate Markdown output
3. Compare with ground-truth
Part 6: Testing Task
Generate Markdown Outputs for:
➢ 3 training images
➢ 3 unseen document images

5. Visualization Module
Display:
• Input image
• Ground truth markdown
• Generated markdown
Deliverables
1. Jupyter Notebook:
• Complete PyTorch implementation.
• Clean, well-structured code
2. Fine Tuning Results
• Training loss
• Validation loss
• Generated markdown outputs
• Comparison (Actual vs Predicted)
3. Screenshots
• Output predictions
• Visualization results
4. App Deployment:
• Build a Gradio or Streamlit App that should:
▪ Upload image
▪ Generate Markdown

Bonus Tasks (Optional)
• Compare 2 epochs vs 5 epochs
• Compare different prompt styles
• Compare zero-shot vs fine-tuned outputs