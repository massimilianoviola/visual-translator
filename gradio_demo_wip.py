import gradio as gr
import torch
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers.image_utils import load_image
import cv2
import numpy as np

# Load the model and processor
model_id = "google/paligemma2-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

def describe_image(image, click_coords):
    """
    Generate a description for the given image.
    If a click coordinate is provided, include it in the prompt.
    """
    if click_coords is not None:

        prompt = f"Identify the object at the tip of the arrow, ignoring the arrow itself; translate into italian"
    else:
        prompt = "describe en"
    
    # Wrap both the text and image in lists
    model_inputs = processor(text=prompt, images=image, return_tensors="pt")
    # Move tensors to the model's device and dtype
    model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)
    
    return decoded

def annotate_image(image, click_coords):
    """
    Annotate the image with an arrow pointing to the clicked location.
    The arrow's starting point is adjusted based on the click's proximity to the image boundaries.
    """
    if image is None:
        return None
    annotated = image.copy()
    if click_coords is not None:
        x = int(click_coords.get("x", 0))
        y = int(click_coords.get("y", 0))
        height, width = annotated.shape[:2]
        offset = 50  # distance for the arrow's starting point

        # Determine the horizontal start point:
        if x - offset < 0:
            start_x = x + offset
        elif x + offset > width:
            start_x = x - offset
        else:
            start_x = x - offset

        # Determine the vertical start point:
        if y - offset < 0:
            start_y = y + offset
        elif y + offset > height:
            start_y = y - offset
        else:
            start_y = y - offset

        start_point = (start_x, start_y)
        end_point = (x, y)
        cv2.arrowedLine(annotated, start_point, end_point, (255, 0, 0), thickness=3, tipLength=0.2)
    return annotated

def update_click(image, evt: gr.SelectData):
    """
    Update the annotation on the image and store the click coordinates.
    The event data (evt.index) is assumed to be a tuple (x, y).
    """
    if evt is not None and evt.index is not None:
        x, y = evt.index  # Extract coordinates from event data
        click_coords = {"x": x, "y": y}
        annotated = annotate_image(image, click_coords)
        return annotated, click_coords
    return image, None

with gr.Blocks() as demo:
    gr.Markdown("# Visual Translator with Click Prompt")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Upload or select an image", interactive=True)
            # A state to hold the click coordinates
            click_coords_state = gr.State(None)
            generate_button = gr.Button("Generate Description")
            description_output = gr.Textbox(label="Generated Description")
        with gr.Column():
            annotated_output = gr.Image(type="numpy", label="Annotated Image")

    # When the image is clicked, update the annotated image and store click coordinates.
    image_input.select(
        update_click,
        inputs=[image_input],
        outputs=[annotated_output, click_coords_state]
    )

    # Clicking the button runs the description function with the current image and click coordinates.
    generate_button.click(
        describe_image,
        inputs=[annotated_output, click_coords_state],
        outputs=description_output
    )

    # Provide example images for quick testing.
    default_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
    ]
    gr.Examples(examples=default_images, inputs=image_input)

demo.launch()