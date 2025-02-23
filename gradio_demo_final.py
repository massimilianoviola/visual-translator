import json

import cv2
import gradio as gr
import torch
from transformers import pipeline

from image_descriptor import PaliGemmaDescriptor, SmolVLMDescriptor


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


def translate_description(descriptions, target_language):
    """
    Translate the input description (word) from English to the target language.
    Generate three example sentences coherent with the context.
    """
    word = descriptions[0]
    context_description = descriptions[1]
    messages = [
        {
            "role": "system",
            "content": """You act as a translator. Provide the output in **strictly valid JSON format**. Ensure it follows this structure exactly:
        {
            "Input language": "English",
            "Output language": "Italian",
            "Context": "Mum is making a cake for the birthday of my sister in the kitchen. They are putting the ingredients together. The cake looks very tasty.",
            "Word to translate": "cake",
            "Translation": "torta",
            "Sentences": [
                "La torta è deliziosa.",
                "Ho mangiato una torta per il compleanno di mia sorella.",
                "La torta sta per essere infornata."
            ]
        }
        Only return valid JSON. Do not include additional commentary.""",
        },
        {
            "role": "user",
            "content": f"""Translate the following word from English to {target_language}: {word}.
            Also return 3 sentences in the output language that use the word and that are coherent with this context: {context_description}\n""",
        },
    ]

    translation_outputs = pipe(messages, max_new_tokens=256)
    # Assume the translation pipeline returns a list of outputs with a 'content' key.
    generated_translation = translation_outputs[0]["generated_text"][-1]
    parsed_translation = json.loads(generated_translation["content"])
    translation_text = (
        f"Translation: {parsed_translation['Translation']}\n"
        "Sentences:\n" + "\n".join(parsed_translation["Sentences"])
    )
    return translation_text


def describe_image_with_model(annotated_image, click_coords, original_image, model_name):
    return descriptors[model_name].describe_image(annotated_image, click_coords, original_image)


# Initialize the image descriptors
smolvlm_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
paligemma_model_id = "google/paligemma2-3b-mix-448"

descriptors = {
    smolvlm_model_id: SmolVLMDescriptor(smolvlm_model_id),
    paligemma_model_id: PaliGemmaDescriptor(paligemma_model_id),
}

for descriptor in descriptors.values():
    descriptor.load_model()

# Initialize the translation pipeline
llm_name = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline("text-generation", model=llm_name, torch_dtype=torch.bfloat16, device_map="auto")

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
                value=list(descriptors.keys())[0],
                label="Select Model",
            )
            describe_button = gr.Button("Generate Description")
            description_output = gr.Textbox(label="Generated Description")
            language_dropdown = gr.Dropdown(
                choices=["German", "French", "Italian", "Portuguese", "Hindi", "Spanish", "Thai"],
                value="Italian",
                label="Select Target Language",
            )
            translate_button = gr.Button("Translate")
            translation_output = gr.Textbox(label="Translation Output")
        with gr.Column():
            annotated_output = gr.Image(type="numpy", label="Annotated Image")

    # When the image is clicked, update the annotated image and store click coordinates
    image_input.select(
        update_click, inputs=[image_input], outputs=[annotated_output, click_coords_state]
    )
    # Add gr.State to store full decoded list
    all_descriptions_state = gr.State()
    describe_button.click(
        describe_image_with_model,
        inputs=[annotated_output, click_coords_state, image_input, model_selector],
        outputs=[description_output, all_descriptions_state],
    )
    translate_button.click(
        translate_description,
        inputs=[all_descriptions_state, language_dropdown],
        outputs=translation_output,
    )
    default_images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/500px-Golde33443.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/500px-Cat03.jpg",
    ]
    gr.Examples(examples=default_images, inputs=image_input)

demo.launch()
