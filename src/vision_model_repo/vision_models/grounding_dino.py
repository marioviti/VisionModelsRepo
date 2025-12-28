from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List

class GroundingDinoInstanceDetection:
    collection = ["IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny"]

    def __init__(self, model_id: str = collection[0], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    @torch.inference_mode()
    def __call__(self, images:List[Image.Image], texts:List[str], box_threshold=0.4, text_threshold=0.3):
        phrases = [t.strip().lower().rstrip(".") + "." for t in texts]
        text = " ".join(phrases)

        inputs = self.processor(images=images, text=text, return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # --- FIX: make text batch match image batch ---
        B = inputs["pixel_values"].shape[0]

        def _repeat_to_batch(x: torch.Tensor, B: int) -> torch.Tensor:
            # repeat along batch dim only
            reps = [B] + [1] * (x.dim() - 1)
            return x.repeat(*reps)

        # These keys are commonly present for GroundingDINO in transformers
        for k in ["input_ids", "attention_mask", "token_type_ids", "text_attention_mask"]:
            if k in inputs and torch.is_tensor(inputs[k]) and inputs[k].shape[0] == 1 and B > 1:
                inputs[k] = _repeat_to_batch(inputs[k], B)
        # ---------------------------------------------

        outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [(img.height, img.width) for img in images],
            device=outputs.logits.device,
        )

        return self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
