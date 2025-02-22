from abc import ABC, abstractmethod
import torch
from accelerate import Accelerator
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor
)

class ImageDescriptor(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.accelerator = Accelerator()

    @abstractmethod
    def load_model(self):
        pass

    def get_prompt(self, click_coords):
        """Get the prompt based on whether coordinates were clicked.
        
        Args:
            click_coords: Dictionary containing x,y coordinates if clicked, None otherwise
            
        Returns:
            str: The prompt to use for image description
        """
        if click_coords is not None:
            return "Identify the object at the tip of the arrow, ignoring the arrow itself\n"
        return "describe en"

    @abstractmethod
    def describe_image(self, annotated_image, click_coords, original_image):
        pass

class SmolVLMDescriptor(ImageDescriptor):
    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self.accelerator.prepare(self.model)

    def describe_image(self, annotated_image, click_coords, original_image):
        """Takes an uploaded image and generates a description using SmolVLM.
        If a click coordinate is provided, include it in the prompt.
        """
        image_to_use = annotated_image if annotated_image is not None else original_image
        prompt = self.get_prompt(click_coords)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image_to_use], return_tensors="pt")
        
        # Handle different input types appropriately
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if key in ["input_ids", "attention_mask"]:  # Text-related tensors
                    inputs[key] = inputs[key].to(dtype=torch.long, device=self.accelerator.device)
                elif key.startswith("pixel"):  # Image-related tensors
                    inputs[key] = inputs[key].to(
                        dtype=torch.float16 if self.accelerator.device.type == "cuda" else torch.float32,
                        device=self.accelerator.device
                    )

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=500)
            generated_texts = self.processor.batch_decode(
                generation,
                skip_special_tokens=True,
            )

            # Remove the prompt from the response
            return generated_texts[0].split("Assistant: ")[-1].strip()

class PaliGemmaDescriptor(ImageDescriptor):
    def load_model(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)

    def describe_image(self, annotated_image, click_coords, original_image):
        """Takes an uploaded image and generates a description using PaliGemma.
        If a click coordinate is provided, include it in the prompt.
        """
        image_to_use = annotated_image if annotated_image is not None else original_image
        prompt = self.get_prompt(click_coords)

        model_inputs = (
            self.processor(text=prompt, images=image_to_use, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded