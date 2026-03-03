"""
Export all observations of one episode from a legacy LeRobot dataset.

The script reads old-format LeRobot files directly:
- `meta/info.json`
- `data/chunk-xxx/episode_xxxxxx.parquet`

For each frame in the selected episode, it:
1. decodes the image observation,
2. overlays `state` and `actions` text at the top,
3. saves the annotated image to disk,
4. prints `state` and `actions` to stdout,
5. writes a JSONL sidecar file with metadata.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one episode obs from a legacy LeRobot dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the legacy LeRobot dataset root (contains meta/ and data/).",
    )
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lerobot_episode_obs"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default=None,
        help="Image feature key in parquet. If omitted, auto-select when there is exactly one image key.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on exported frames. By default (unset), export all frames in the episode.",
    )
    parser.add_argument("--font-path", type=str, default=None, help="Optional TTF font path for overlay text.")
    parser.add_argument("--font-size", type=int, default=12, help="Overlay font size.")
    parser.add_argument(
        "--line-height",
        type=int,
        default=0,
        help="Overlay line height in pixels. Use <=0 to auto-compute from the font.",
    )
    parser.add_argument("--padding", type=int, default=8, help="Padding for overlay area in pixels.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(data)}")
    return data


def find_image_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise KeyError("`features` is missing or invalid in meta/info.json")

    image_keys: list[str] = []
    for key, spec in features.items():
        if isinstance(spec, dict) and spec.get("dtype") == "image":
            image_keys.append(key)
    return image_keys


def resolve_episode_parquet(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path")
    if not isinstance(template, str):
        raise KeyError("`data_path` is missing or invalid in meta/info.json")

    chunk_size_raw = info.get("chunks_size", 1000)
    chunk_size = int(chunk_size_raw)
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunks_size={chunk_size}")

    episode_chunk = episode_index // chunk_size
    rel_path = template.format(episode_chunk=episode_chunk, episode_index=episode_index)
    candidate = dataset_dir / rel_path
    if candidate.exists():
        return candidate

    fallback = dataset_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Episode parquet not found for episode_index={episode_index}. Tried: {candidate} and {fallback}"
    )


def load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default(size=font_size)
    font_file = Path(font_path)
    if not font_file.exists():
        raise FileNotFoundError(f"Font file not found: {font_file}")
    return ImageFont.truetype(str(font_file), size=font_size)


def format_vector(values: Any) -> str:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.array2string(array, precision=4, separator=", ", suppress_small=False, max_line_width=10_000)


def decode_image_cell(image_cell: Any, dataset_dir: Path) -> Image.Image:
    image_bytes: bytes | None = None
    image_path: str | None = None

    if isinstance(image_cell, dict):
        bytes_obj = image_cell.get("bytes")
        path_obj = image_cell.get("path")
        if isinstance(bytes_obj, bytes | bytearray):
            image_bytes = bytes(bytes_obj)
        if isinstance(path_obj, str):
            image_path = path_obj
    elif isinstance(image_cell, bytes | bytearray):
        image_bytes = bytes(image_cell)
    elif isinstance(image_cell, str):
        image_path = image_cell
    else:
        raise TypeError(f"Unsupported image cell type: {type(image_cell)}")

    if image_bytes:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    if image_path:
        path = Path(image_path)
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as image:
            return image.convert("RGB")

    raise ValueError("Image cell contains neither bytes nor path")


def overlay_text(
    image: Image.Image,
    lines: list[str],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    line_height: int,
    padding: int,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)

    final_line_height = line_height
    if final_line_height <= 0:
        bbox = draw.textbbox((0, 0), "Ag", font=font)
        final_line_height = (bbox[3] - bbox[1]) + 4

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        y += final_line_height

    return output


def main() -> None:
    args = parse_args()
    if args.episode_index < 0:
        raise ValueError("--episode-index must be >= 0")

    dataset_dir = args.dataset_dir.resolve()
    info_path = dataset_dir / "meta" / "info.json"
    info = load_json(info_path)

    total_episodes = int(info.get("total_episodes", 0))
    if total_episodes and args.episode_index >= total_episodes:
        raise ValueError(f"--episode-index={args.episode_index} out of range (total_episodes={total_episodes})")

    image_keys = find_image_keys(info)
    if args.image_key is None:
        if not image_keys:
            raise ValueError("No image features found in meta/info.json")
        if len(image_keys) > 1:
            raise ValueError(f"Multiple image keys found {image_keys}, please specify --image-key")
        image_key = image_keys[0]
    else:
        image_key = args.image_key
        if image_key not in image_keys:
            raise ValueError(f"--image-key={image_key} not in available image keys: {image_keys}")

    parquet_path = resolve_episode_parquet(dataset_dir, info, args.episode_index)
    table = pq.read_table(parquet_path)

    required_columns = {image_key, "state", "actions"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise KeyError(f"Missing required columns in parquet: {sorted(missing_columns)}")

    rows = table.to_pylist()
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be > 0 when set")
        rows = rows[: args.max_frames]

    output_episode_dir = args.output_dir.resolve() / f"episode_{args.episode_index:06d}"
    output_episode_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_episode_dir / "state_actions.jsonl"

    font = load_font(args.font_path, args.font_size)

    exported = 0
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for local_frame_id, row in enumerate(rows):
            image = decode_image_cell(row[image_key], dataset_dir)
            state = row["state"]
            actions = row["actions"]

            frame_index = int(row.get("frame_index", local_frame_id))
            timestamp = float(row.get("timestamp", 0.0))
            state_text = format_vector(state)
            action_text = format_vector(actions)

            lines = [
                f"frame_index={frame_index} timestamp={timestamp:.3f}",
                f"state={state_text}",
                f"action={action_text}",
            ]

            annotated = overlay_text(
                image,
                lines,
                font,
                line_height=args.line_height,
                padding=args.padding,
            )
            frame_file = output_episode_dir / f"frame_{local_frame_id:06d}.png"
            annotated.save(frame_file)

            print(f"[{local_frame_id:04d}] state={state_text} action={action_text}")

            record = {
                "episode_index": args.episode_index,
                "frame_local_index": local_frame_id,
                "frame_index": frame_index,
                "timestamp": timestamp,
                "state": np.asarray(state, dtype=np.float32).reshape(-1).tolist(),
                "actions": np.asarray(actions, dtype=np.float32).reshape(-1).tolist(),
                "image_file": frame_file.name,
            }
            jsonl_file.write(json.dumps(record) + "\n")
            exported += 1

    print(f"Saved {exported} frames to: {output_episode_dir}")
    print(f"Saved state/action JSONL to: {jsonl_path}")


if __name__ == "__main__":
    main()
