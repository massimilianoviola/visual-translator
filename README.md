# Visual Translator Demo

Welcome to the **Visual Translator Demo**! This project is part of the **Entrepreneur First Hackathon 2025** in Zurich. Our goal is to create a translation companion that enhances the way people interact with their environment, enabling them to translate objects they can see and capture in real-time.

By simply uploading or selecting an image and clicking on the object you want to observe and translate, you’ll receive not only a translation but also a description, with sample sentences to help you better understand the context of the object.

## 🚀 Features

- **Upload or Select Images**: Choose an image from your computer or use one of the default examples to get started.
- **Click to Annotate**: Click on the object you want to translate, and the system will annotate the image with an arrow.
- **Model Selection**: Choose from two state-of-the-art translation models, SmolVLM and PaliGemma.
- **Translation and Descriptions**: Receive the translation of the object along with a description and three sample sentences for better context.
- **Gradio Interface**: A user-friendly interface that makes it easy to interact with the tool.

## 🛠 Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-repository/gradio_demo
    cd gradio_demo
    ```

2. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up your ElevenLabs API key**:
    Create a `.env` file in the root directory and add your API key:

    ```
    ELEVENLABS_API_KEY=your_api_key_here
    ```

4. **Run the demo**:

    ```bash
    python gradio_demo.py
    ```

## 📦 Requirements

- `accelerate`
- `gradio`
- `opencv-python`
- `torch`
- `transformers` (from GitHub)
- `num2words`
- `dotenv`
- `elevenlabs`

## 🌍 Usage

1. Upload an image or choose one from the example options.
2. Click on the image to place an arrow on the object you wish to translate.
3. Select the model you prefer (SmolVLM or PaliGemma).
4. Click the "Generate Description" button to get the translated description and sample sentences.

## 🤝 Team Members

This project is brought to you by the talented team members participating in the **Entrepreneur First Hackathon 2025**, Zurich:

- **Massimiliano Viola**
- **You Wu**
- **Timur Taepov**

## 🔑 Notes

- This demo uses two pre-trained models: **SmolVLM** and **PaliGemma**.
- Ensure that your ElevenLabs API key is correctly added to the `.env` file before running the application.
