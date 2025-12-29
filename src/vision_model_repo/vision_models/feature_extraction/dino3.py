import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List

class DinoV3:
    """
    DINO3 (Deep Neural Network for Instance Segmentation) is a state-of-the-art instance segmentation model.
    """

    collection = [ "facebook/dinov3-convnext-tiny-pretrain-lvd1689m", 
                   "facebook/dinov3-convnext-large-pretrain-lvd1689m",
                   "facebook/dinov3-vith16plus-pretrain-lvd1689m" ]

    def __init__(self, model: str = collection[0], device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval().to(device)

    @torch.inference_mode()
    def __call__(self, images:List[Image.Image]):
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return outputs