import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    """Converts model actions back to Franka action space."""

    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        # Keep [x, y, z, rx, ry, rz, gripper].
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class FrankaEEInputs(transforms.DataTransformFn):
    """Converts Franka dataset fields into model input format."""

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0
    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (7,), (
            f"Expected state shape (7,), got {data['observation/state'].shape}"
        )
        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(data["observation/state"]).float()

        state = data["observation/state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])

        # We only mask padding for pi0 model, not pi0-FAST.
        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
            image_masks = (np.True_, np.False_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 7, (
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            )
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
