import json
import os

import cv2
import gradio as gr
import torch
from elevenlabs.client import ElevenLabs
from transformers import pipeline

from image_descriptor import PaliGemmaDescriptor, SmolVLMDescriptor


def draw_arrow(image, click_coords, arrow_color=(255, 0, 0), thickness=15, tip_length=0.25):
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
    return (
        parsed_translation["Translation"],
        parsed_translation["Sentences"][0],
        parsed_translation["Sentences"][1],
        parsed_translation["Sentences"][2],
    )


def describe_image_with_model(annotated_image, click_coords, original_image, model_name):
    return descriptors[model_name].describe_image(annotated_image, click_coords, original_image)


def vocalize_text(text):
    """
    Converts the given text into speech using the ElevenLabs API.
    """
    password = os.getenv("ELEVENLABS_API_KEY")
    if password is None:
        gr.Warning("ELEVENLABS_API_KEY is not set. Please configure your API key.")
        return

    client = ElevenLabs(
        api_key=password,
    )

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="iP95p4xoKVk53GoZ742B",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    # Read all bytes from the generator
    audio = b"".join(chunk for chunk in audio)

    return audio


def vocalize_all_text(translation, sentence1, sentence2, sentence3):
    """
    Vocalize all text inputs at once and return audio for each.
    """
    return (
        vocalize_text(translation),
        vocalize_text(sentence1),
        vocalize_text(sentence2),
        vocalize_text(sentence3),
    )


# Initialize the image descriptors
smolvlm_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
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
            sentence1_output = gr.Textbox(label="Example Sentence 1")
            sentence2_output = gr.Textbox(label="Example Sentence 2")
            sentence3_output = gr.Textbox(label="Example Sentence 3")
        with gr.Column():
            annotated_output = gr.Image(type="numpy", label="Annotated Image")
            vocalize_all_button = gr.Button("Vocalize All")
            word_audio = gr.Audio(label="Word Audio", format="mp3")
            sentence1_audio = gr.Audio(label="Sentence 1 Audio", format="mp3")
            sentence2_audio = gr.Audio(label="Sentence 2 Audio", format="mp3")
            sentence3_audio = gr.Audio(label="Sentence 3 Audio", format="mp3")

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
        outputs=[translation_output, sentence1_output, sentence2_output, sentence3_output],
    )
    vocalize_all_button.click(
        vocalize_all_text,
        inputs=[translation_output, sentence1_output, sentence2_output, sentence3_output],
        outputs=[word_audio, sentence1_audio, sentence2_audio, sentence3_audio],
    )
    default_images = [
        "./assets/realworldQA_25.jpg",
        "./assets/realworldQA_48.jpg",
        "./assets/realworldQA_378.jpg",
        "./assets/realworldQA_499.jpg",
        "./assets/realworldQA_503.jpg",
        "./assets/realworldQA_546.jpg",
        "./assets/realworldQA_564.jpg",
        "./assets/realworldQA_724.jpg",
        "./assets/realworldQA_760.jpg",
    ]
    gr.Examples(examples=default_images, inputs=image_input)

demo.launch()
