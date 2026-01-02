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

    def _resolve_overlaps(self, grouped_results: List[List[Dict[str, Any]]], size: Tuple[int, int]):
        """
        Resolves overlaps for a single image's results using pixel-wise max score.
        grouped_results: List of dicts for one image, each having 'result' with 'masks' (N, H, W) and 'iou_scores' (N,)
        """
        # Collect all instances
        all_masks = []
        all_scores = []
        instance_map = [] # (prompt_idx, instance_idx_within_prompt)

        for p_idx, res in enumerate(grouped_results):
             masks = res["result"].get("masks") # (N, H, W)
             scores = res["result"].get("iou_scores") # (N,)
             
             if masks is None or scores is None: continue
             
             # ensure on CPU
             masks = masks.detach().cpu()
             scores = scores.detach().cpu()
             
             for i in range(masks.shape[0]):
                 all_masks.append(masks[i])
                 all_scores.append(scores[i])
                 instance_map.append((p_idx, i))

        if not all_masks:
            return

        # Stack: (Total_instances, H, W)
        stack_masks = torch.stack(all_masks)
        stack_scores = torch.stack(all_scores) # (Total_instances,)

        # Create a score map: (Total_instances, H, W) - broadcast score to mask area
        # We only care about scores where mask is True
        # But to use argmax, let's just use the score value.
        
        # (Total_instances, H, W)
        score_map = torch.zeros_like(stack_masks, dtype=torch.float32)
        
        # Assign scores to active pixels
        for i in range(len(stack_scores)):
            score_map[i, :, :] = torch.where(stack_masks[i], stack_scores[i], 0.0)

        # Find winner index for each pixel: (H, W) values in [0, Total_instances-1]
        # We need to handle background (where all are 0).
        # max_scores: (H, W), indices: (H, W)
        max_scores, best_indices = torch.max(score_map, dim=0)
        
        # Identify pixels that have at least one mask
        any_mask = torch.any(stack_masks, dim=0)
        
        # Now update each mask to be True ONLY if it is the winner AND the pixel was active
        for i in range(len(stack_masks)):
            # Pixel belongs to instance i if:
            # 1. It was claimed by i originally (stack_masks[i]) -- implicit in scores
            # 2. i is the best index (best_indices == i)
            # 3. The pixel had any mask (any_mask) - ensures we don't pick up background zeros if scores are weird
            
            # Actually simple: mask[i] = (best_indices == i) & (max_scores > 0)
            
            new_mask = (best_indices == i) & (max_scores > 0)
            
            # Update the original tensor in the list
            p_idx, inst_idx = instance_map[i]
            
            # We need to modify the tensor inside the grouped_results
            # The tensor is (N, H, W), so we update slice [inst_idx]
            # Warning: we detached earlier. We must update the original if we want to return it.
            # Or better, we constructed this from detached CPUs. We should write back.
            
            # Since 'grouped' holds the result dicts, let's update them.
            # But the result dicts hold the original tensors which might be on GPU.
            # We should probably do this logic on the device of the tensors to avoid transfers if possible,
            # but user didn't specify. Assuming mixed usage, let's stick to what we have or move to device.
            
            # Let's perform updates on the original device if possible, or just overwrite with CPU tensor.
            # Overwriting with CPU tensor is safer for memory.
            
            grouped_results[p_idx]["result"]["masks"][inst_idx] = new_mask.to(grouped_results[p_idx]["result"]["masks"].device)


    @torch.inference_mode()
    def __call__(
        self,
        images: List[Image.Image],
        texts: TextBatch,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        resolve_overlaps: bool = False,
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

        if resolve_overlaps:
            for bi in range(len(images)):
                if not grouped[bi]: continue
                # We assume all images in batch have same size if processed together, 
                # but target_sizes has them. Let's get size from one of the masks or the image.
                # Masks are resized to original size by post_process_instance_segmentation
                
                # Check first valid result
                h, w = target_sizes[bi]
                self._resolve_overlaps(grouped[bi], (h, w))

        return grouped
