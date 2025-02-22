import cv2
import gradio as gr
from image_descriptor import SmolVLMDescriptor, PaliGemmaDescriptor

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

# Initialize the image descriptors
smolvlm_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
paligemma_model_id = "google/paligemma2-3b-mix-448"

descriptors = {
    smolvlm_model_id: SmolVLMDescriptor(smolvlm_model_id),
    paligemma_model_id: PaliGemmaDescriptor(paligemma_model_id)
}

# Load both models
for descriptor in descriptors.values():
    descriptor.load_model()

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
            
            # Add model selector dropdown
            model_selector = gr.Dropdown(
                choices=list(descriptors.keys()),
                value=list(descriptors.keys())[0], # Use first model as default instead of hardcoding
                label="Select Model"
            )
            
            generate_button = gr.Button("Generate Description")
            description_output = gr.Textbox(label="Generated Description")
        with gr.Column():
            annotated_output = gr.Image(type="numpy", label="Annotated Image")

    # When the image is clicked, update the annotated image and store click coordinates
    image_input.select(
        update_click, inputs=[image_input], outputs=[annotated_output, click_coords_state]
    )

    # Clicking the button runs the description function with the current image and click coordinates
    def describe_image_with_model(annotated_image, click_coords, original_image, model_name):
        return descriptors[model_name].describe_image(annotated_image, click_coords, original_image)

    generate_button.click(
        describe_image_with_model,
        inputs=[annotated_output, click_coords_state, image_input, model_selector],
        outputs=description_output,
    )

    default_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
    ]
    gr.Examples(examples=default_images, inputs=image_input)

demo.launch()
