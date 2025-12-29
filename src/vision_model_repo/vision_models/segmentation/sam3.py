import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image
from typing import List, Union, Tuple, Dict, Any

TextBatch = Union[List[str], List[List[str]]]

class Sam3:
    collection = ["facebook/sam3"]

    def __init__(self, model_id: str = collection[0], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_id).eval().to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_id)

    def _expand_pairs(
        self, images: List[Image.Image], texts: TextBatch
    ) -> Tuple[List[Image.Image], List[str], List[Tuple[int, int]]]:
        """
        Returns:
          flat_images: repeated images
          flat_texts:  one prompt per entry
          index_map:   (image_index, prompt_index) for each flat entry
        """
        B = len(images)
        if B == 0:
            raise ValueError("images is empty")

        # Case A: ["sauce","meat"] => prompts shared across all images (no join)
        if len(texts) > 0 and isinstance(texts[0], str):
            prompts: List[str] = [t.strip() for t in texts]
            flat_images, flat_texts, index_map = [], [], []
            for bi, img in enumerate(images):
                for pi, p in enumerate(prompts):
                    flat_images.append(img)
                    flat_texts.append(p)
                    index_map.append((bi, pi))
            return flat_images, flat_texts, index_map

        # Case B: [["sauce","meat"], ["rice"]] => per-image prompts (no join)
        if len(texts) != B:
            raise ValueError(f"texts must be length {B} when nested; got {len(texts)}")

        flat_images, flat_texts, index_map = [], [], []
        for bi, per_img_prompts in enumerate(texts):
            if not per_img_prompts:
                continue
            for pi, p in enumerate(per_img_prompts):
                flat_images.append(images[bi])
                flat_texts.append(p.strip())
                index_map.append((bi, pi))

        if not flat_images:
            raise ValueError("No prompts provided after normalization")

        return flat_images, flat_texts, index_map

    @torch.inference_mode()
    def __call__(
        self,
        images: List[Image.Image],
        texts: TextBatch,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[List[Dict[str, Any]]]:

        flat_images, flat_texts, index_map = self._expand_pairs(images, texts)

        inputs = self.processor(images=flat_images, text=flat_texts, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Must align with flattened batch (one per (image,prompt))
        if "original_sizes" in inputs:
            target_sizes = inputs["original_sizes"].tolist()
        else:
            target_sizes = [(img.height, img.width) for img in flat_images]

        flat_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )

        # regroup: per original image, list of prompt results
        grouped: List[List[Dict[str, Any]]] = [[] for _ in images]
        for res, (bi, pi), prompt in zip(flat_results, index_map, flat_texts):
            grouped[bi].append({"prompt_index": pi, "prompt": prompt, "result": res})

        return grouped
