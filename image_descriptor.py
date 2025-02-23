from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)


class ImageDescriptor(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        pass

    def get_prompts(self, click_coords):
        """Get the prompts based on whether coordinates are provided.
        If so, the first prompt is used to identify the object at the tip of the arrow.
        The second prompt is used to generate a global image caption.
        """
        if click_coords is not None:
            prompts = [
                "Identify the object at the tip of the arrow. Answer only with the object name and nothing else.\n",
                "caption en\n",
            ]
        else:
            prompts = ["caption en\n", "caption en\n"]
        return prompts

    @abstractmethod
    def describe_image(self, annotated_image, click_coords, original_image):
        pass


class SmolVLMDescriptor(ImageDescriptor):
    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def describe_image(self, annotated_image, click_coords, original_image):
        """Take an uploaded image and generate a description using SmolVLM2.
        If a click coordinate is provided, include it in the prompt.
        """
        prompts = self.get_prompts(click_coords)
        if annotated_image is not None:
            images_list = [annotated_image, original_image]
        else:
            images_list = [original_image, original_image]

        responses = []
        for i in range(2):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompts[i]},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = (
                self.processor(text=prompt, images=images_list[i], return_tensors="pt")
                .to(torch.bfloat16)
                .to(self.model.device)
            )

            with torch.inference_mode():
                generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                generated_text = self.processor.decode(generation[0], skip_special_tokens=True)
                # Remove the prompt from the response
                responses.append(generated_text.split("Assistant: ")[-1].strip())

        return responses[0], responses


class PaliGemmaDescriptor(ImageDescriptor):
    def load_model(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)

    def describe_image(self, annotated_image, click_coords, original_image):
        """Take an uploaded image and generate a description using PaliGemma.
        If a click coordinate is provided, include it in the prompt.
        """
        prompts = self.get_prompts(click_coords)
        if annotated_image is not None:
            images_list = [annotated_image, original_image]
        else:
            images_list = [original_image, original_image]

        model_inputs = (
            self.processor(text=prompts, images=images_list, return_tensors="pt", padding=True)
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        # Just take the first input length since they're all padded to the same length
        input_len = model_inputs["input_ids"][0].shape[-1]

        with torch.inference_mode():
            generations = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            decoded = [
                self.processor.decode(gen[input_len:], skip_special_tokens=True).strip()
                for gen in generations
            ]
        return decoded[0], decoded
