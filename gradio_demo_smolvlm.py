import cv2
import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator

def draw_arrow(image, click_coords, arrow_color=(255, 0, 0), thickness=5, tip_length=0.25):
    """Annotate the image with an arrow pointing to the clicked location.
    The dimension of the arrow is computed proportionally to the image size.
    """
    if image is None:
        return None

    annotated = image.copy()

    if click_coords:
        x, y = int(click_coords["x"]), int(click_coords["y"])
        height, width = annotated.shape[:2]

        # Set offset as 15% of the minimum image dimension
        offset = int(min(width, height) * 0.15)

        # Determine the starting point based on click position relative to center
        start_x = x - offset if x > width // 2 else x + offset
        start_y = y - offset if y > height // 2 else y + offset

        # Draw the arrow
        start_point, end_point = (start_x, start_y), (x, y)
        cv2.arrowedLine(
            annotated, start_point, end_point, arrow_color, thickness, tipLength=tip_length
        )

    return annotated

def update_click(image, evt: gr.SelectData):
    """
    Update the arrow on the image and store the click coordinates.
    The event data (evt.index) is assumed to be a tuple (x, y).
    """
    if evt is not None and evt.index is not None:
        x, y = evt.index  # Extract coordinates from event data
        click_coords = {"x": x, "y": y}
        annotated = draw_arrow(image, click_coords)
        return annotated, click_coords
    return image, None


def describe_image(annotated_image, click_coords, original_image):
    """Takes an uploaded image and generates a description using SmolVLM.
    If a click coordinate is provided, include it in the prompt.
    """
    image_to_use = annotated_image if annotated_image is not None else original_image
    if click_coords is not None:
        prompt = "Identify the object at the tip of the arrow, ignoring the arrow itself\n"
    else:
        prompt = "describe en"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image_to_use], return_tensors="pt")
    
    # Handle different input types appropriately
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key in ["input_ids", "attention_mask"]:  # Text-related tensors
                inputs[key] = inputs[key].to(dtype=torch.long, device=accelerator.device)
            elif key.startswith("pixel"):  # Image-related tensors
                inputs[key] = inputs[key].to(
                    dtype=torch.float16 if accelerator.device.type == "cuda" else torch.float32,
                    device=accelerator.device
                )

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(
            generation,
            skip_special_tokens=True,
        )

        # Remove the prompt from the response
        return generated_texts[0].split("Assistant: ")[-1].strip()

# Initialize the accelerator
accelerator = Accelerator()

# Load the model and processor with device mapping
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.float16 if accelerator.device.type == "cuda" else torch.float32,  # Use float16 for GPU, float32 for CPU/MPS
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Prepare the model with the accelerator
model = accelerator.prepare(model)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Visual Translator")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="numpy", label="Upload or select an image", interactive=True
            )
            # A state to hold the click coordinates
            click_coords_state = gr.State(None)
            generate_button = gr.Button("Generate Description")
            description_output = gr.Textbox(label="Generated Description")
        with gr.Column():
            annotated_output = gr.Image(type="numpy", label="Annotated Image")

    # When the image is clicked, update the annotated image and store click coordinates
    image_input.select(
        update_click, inputs=[image_input], outputs=[annotated_output, click_coords_state]
    )

    # Clicking the button runs the description function with the current image and click coordinates
    generate_button.click(
        describe_image,
        inputs=[annotated_output, click_coords_state, image_input],
        outputs=description_output,
    )

    default_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
    ]
    gr.Examples(examples=default_images, inputs=image_input)

demo.launch()
