import gradio as gr
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image

# Load the model and processor
model_id = "google/paligemma2-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

def describe_image(image):
    """Takes an uploaded image and generates a description using PaliGemma."""
    prompt = "describe en"
    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    return decoded

default_images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
]

default_image_objects = [load_image(url) for url in default_images]

# Gradio interface
demo = gr.Interface(
    fn=describe_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Generated Description"),
    title="Visual Translator",
    description="Upload an image, and the model will describe it in natural language.",
    examples=default_images
)

demo.launch()
