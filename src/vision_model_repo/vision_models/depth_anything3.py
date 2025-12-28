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

    def __call__(self, images: list[Image.Image], export_format="glb", **options) -> list[dict]:
        prediction = self.predict(images, export_format=export_format, **options)
        # return list of datum with a prediction in each (list of datum of dict type
        # prediction.depth: [N,H,W], conf: [N,H,W], intrinsics: [N,3,3], extrinsics: [N,3,4]
        depth = prediction.depth
        conf = prediction.conf
        intrinsics = prediction.intrinsics
        extrinsics = prediction.extrinsics

        n = int(depth.shape[0])
        out = []
        for i in range(n):
            payload = {
                    "kind": "depth_anything_v3",
                    "units": "m",
                    "index": i,
                    "model": self.model_id,
                    "conf" : conf[i] if conf is not None else None,
                    "depth": depth[i],
                    "intrinsics": intrinsics[i] if intrinsics is not None else None,
                    "extrinsics": extrinsics[i] if extrinsics is not None else None,
                }
            out.append(payload)

        return out

    @torch.inference_mode()
    def predict(self, images: list[Image.Image], export_format="glb", **options):
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