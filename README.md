# Visual Translator Demo

This repository contains a demo of a visual translation companion, designed to help translate objects or scenes that a person can see and capture in a picture. By clicking on the image and selecting an object, you can receive translations and descriptions based on the selected object, with sample sentences for better understanding.

## Features

- Upload or select an image to annotate.
- Draw arrows on the image by clicking on the objects to observe and translate.
- Choose between two translation models for generating object descriptions.
- Get three sentence samples for each description.
- Interact with the image through Gradio UI for ease of use.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-repository/gradio_demo
    cd gradio_demo
    ```

2. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your ElevenLabs API key to the `.env` file:

    ```
    ELEVENLABS_API_KEY=your_api_key_here
    ```

4. Run the demo:

    ```bash
    python gradio_demo.py
    ```

## Requirements

- `accelerate`
- `gradio`
- `opencv-python`
- `torch`
- `transformers` (from GitHub)
- `num2words`
- `dotenv`
- `elevenlabs`

## Usage

1. Upload an image or select one from the example options.
2. Click on the image to place an arrow on the object you want to observe and translate.
3. Select a translation model from the dropdown.
4. Press "Generate Description" to get the translated description and sample sentences.

## Notes

- This tool works with two pre-trained models: SmolVLM and PaliGemma.
- Make sure your ElevenLabs API key is valid and added to the `.env` file.

