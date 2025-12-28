import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
from typing import List, Union

TextBatch = Union[List[str], List[List[str]]]

class Sam3:
    
    collection = ["facebook/sam3"]

    def __init__(self, model_id: str = collection[0], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).eval().to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)

    def _normalize_texts(self, images: List[Image.Image], texts: TextBatch):
        B = len(images)

        # Case A: ["sauce", "meat"]  -> shared prompts for every image
        if len(texts) > 0 and isinstance(texts[0], str):
            # If user provided a single prompt -> broadcast
            if len(texts) == 1 and B > 1:
                return [texts[0]] * B
            # If user provided B prompts -> per-image
            if len(texts) == B:
                return texts
            # Otherwise treat as shared multi-prompt: join them into one string per image
            joined = ", ".join([t.strip() for t in texts])
            return [joined] * B

        # Case B: [["sauce","meat"], ["rice"]] -> per-image list of prompts
        if len(texts) == 1 and B > 1:
            # broadcast list-of-prompts to each image by joining
            joined = ", ".join([t.strip() for t in texts[0]])
            return [joined] * B

        if len(texts) != B:
            raise ValueError(f"texts must be length 1 or {B}, got {len(texts)}")

        # join per-image lists into per-image strings
        return [", ".join([t.strip() for t in per_img]) for per_img in texts]

    @torch.inference_mode()
    def __call__(
        self,
        images: List[Image.Image],
        texts: TextBatch,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        texts = self._normalize_texts(images, texts)

        inputs = self.processor(images=images, text=texts, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # processor usually gives original_sizes; keep fallback just in case
        if "original_sizes" in inputs:
            target_sizes = inputs["original_sizes"].tolist()
        else:
            target_sizes = [(img.height, img.width) for img in images]

        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )
