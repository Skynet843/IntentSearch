from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests

class BLIPCaptionGenerator:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cpu"):
        """
        Initialize the BLIP model and processor.
        
        Args:
            model_name (str): Hugging Face model identifier.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    def generate_caption(self, image_path_or_url, conditional_prompt=None, max_new_tokens=100):
        """
        Generate a caption for an image.

        Args:
            image_path_or_url (str): Local file path or image URL.
            conditional_prompt (str, optional): Optional prompt like "a photo of".
            max_new_tokens (int): Max tokens to generate.

        Returns:
            str: Generated caption.
        """
        # Load image
        if image_path_or_url.startswith("http") or image_path_or_url.startswith("https"):
            image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        # Prepare inputs
        if conditional_prompt:
            inputs = self.processor(image, conditional_prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Generate caption
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return caption