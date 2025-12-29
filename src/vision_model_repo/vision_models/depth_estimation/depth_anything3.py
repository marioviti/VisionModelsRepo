import torch
from depth_anything_3.api import DepthAnything3
from PIL import Image
from typing import List

class DepthAnythingV3:

    collection = [ 
        "depth-anything/da3-base",
        "depth-anything/da3-small",
        "depth-anything/da3metric-large", 
        "depth-anything/da3nested-giant-large"
    ]

    def __init__(self, model_id: str = collection[0], device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize depth estimation pipeline."""
        self.device = torch.device(device)
        model = DepthAnything3.from_pretrained(model_id)
        model = model.to(device=self.device)
        self.model = model
        self.model_id = model_id

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image], export_format="glb", **options):
        """
        Estimate depth from an image.

        Args:
            image (PIL.Image): Input RGB image.

        Returns:
            PIL.Image: Depth map resized to the same size as the input.
        """
        model = self.model
        prediction = model.inference(
            images,
            export_format=export_format  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
        )

        # Access results
        #depth = prediction.depth         # Depth maps: [N, H, W] float32
        #conf = prediction.conf               # Confidence maps: [N, H, W] float32
        #extrinsics = prediction.extrinsics   # Camera poses (w2c): [N, 3, 4] float32
        #intrinsics = prediction.intrinsics   # Camera intrinsics: [N, 3, 3] float32

        return prediction