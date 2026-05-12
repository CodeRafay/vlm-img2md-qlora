import streamlit as st
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from qwen_vl_utils import process_vision_info
import os

# Streamlit App Configuration
st.set_page_config(page_title="VLM Markdown Generator", layout="wide")

# Path where the model is saved (adjust this if your OUTPUT_DIR was different)
MODEL_DIR = "qwen2-vl-nougat-qlora-final"

@st.cache_resource
def load_model():
    """
    Loads the fine-tuned Qwen2-VL model and processor. 
    Uses @st.cache_resource to load only once per session.
    """
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found at `{MODEL_DIR}`.\\n\\nPlease ensure you have run the notebook to save the model weights first.")
        st.stop()
    
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    
    # 1. Load the original base model
    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 2. Apply the LoRA adapter weights on top of the base model
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    
    return processor, model

st.title("VLM Markdown Generator (Qwen2-VL)")
st.write("Upload a document image and the fine-tuned model will generate its Markdown representation.")

# Load the model
with st.spinner("Loading the fine-tuned model..."):
    processor, model = load_model()

# Image Uploader
uploaded_file = st.file_uploader("Choose a document image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Generated Markdown")
        
        if st.button("Generate Markdown"):
            with st.spinner("Analyzing image and generating markdown... This may take a few moments."):
                
                # Create the input messages exactly as in the training pipeline
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text", 
                                "text": "Convert this document image into well-structured Markdown format. Preserve all headings, paragraphs, lists, tables, equations, and code blocks."
                            }
                        ]
                    }
                ]
                
                # Apply chat template
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                
                # Prepare inputs for the model
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # Generate output
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                
                # Extract only the generated part (ignore the prompt tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # Decode the output
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                # Display the rendered markdown and raw code
                st.markdown(output_text)
                
                st.write("---")
                st.subheader("Raw Markdown Code")
                st.code(output_text, language="markdown")
