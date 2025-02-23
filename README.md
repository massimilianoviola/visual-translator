# Visual Translator 🖼️📣

Welcome to the **Visual Translator**! This project is part of the **Entrepreneur First Hackathon 2025** in Zurich. Our goal is to create a translation companion that enhances the way people interact with their environment, enabling them to translate objects they can see and capture in real-time.

By simply uploading or selecting an image and clicking on the object you want to observe and translate, you’ll receive not only a translation but also a description, with sample sentences to help you better understand the context of the object.

## 🚀 Features

- **Upload or Select Images**: Choose an image from your computer or use one of the default examples to get started.
- **Click to Annotate**: Click on the object you want to translate, and the system will annotate the image with an arrow.
- **Model Selection**: Choose from two vision-language models, SmolVLM2 and PaliGemma 2 mix.
- **Translation and Descriptions**: Receive the translation of the object along with a description and three sample sentences for better context.
- **Gradio Interface**: A user-friendly interface that makes it easy to interact with the tool.

## 🛠 Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/massimilianoviola/visual-translator.git
    ```

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the demo** (an ElevenLabs API key is required):

    ```bash
    ELEVENLABS_API_KEY=YOUR_API_KEY python gradio_demo.py
    ```

## 🌍 Usage

1. Upload an image or choose one from the example options.
2. Click on the image to place an arrow on the object you wish to translate.
3. Select the vision-language model you prefer.
4. Click the "Generate Description" button to get the translated description and sample sentences.
5. Click the "Translate" button to get the translation of the object and the sample sentences.
5. Click the "Vocalize All" button to get the audio of the translation and sample sentences.

## 🤝 Team Members

This project is brought to you by the talented team members participating in the **Entrepreneur First Hackathon 2025**, Zurich:

- **Massimiliano Viola**
- **You Wu**
- **Timur Taepov**

## 🔑 Notes

- This demo uses two pre-trained models: **SmolVLM2** and **PaliGemma 2 mix**.
- Ensure that your ElevenLabs API key is passed as an environment variable to the demo.
