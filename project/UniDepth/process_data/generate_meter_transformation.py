"""Recover metric camera translation from reference and predicted depth."""

import os
import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from unidepth.models import UniDepthV2


def parse_args():
    """Parse command-line options for this processing stage.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="UniDepth video depth estimation")
    parser.add_argument(
        "--reference_root",
        type=str,
        required=True,
        help="Reference HDF5 directory containing frame_path, depth and transformation"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for HDF5 files"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/unidepth-v2-vitl14",
        help="UniDepth model directory"
    )
    parser.add_argument(
        "--min_valid_ratio",
        type=float,
        default=0.9,
        help="Minimum successful-frame fraction; skip videos below it (default: 0.9)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Inclusive first video index for multi-GPU processing"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Inclusive last video index for multi-GPU processing"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU identifier for logging only"
    )
    return parser.parse_args()


def load_model(model_path: str, device: torch.device) -> UniDepthV2:
    """Load the pretrained model and select evaluation mode.

    Args:
        model_path (str): Model checkpoint directory, relative to the release root.
        device (torch.device or str): Device used for inference.

    Returns:
        UniDepthV2: The pretrained model on the requested device.
    """
    model = UniDepthV2.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model


def get_hdf5_files(reference_root: str, start: int = None, end: int = None) -> list:
    """List sorted HDF5 paths within an inclusive range.

    Args:
        reference_root (str): Directory containing VGGT reference HDF5 files.
        start (int or None): Inclusive first index in sorted input order.
        end (int or None): Inclusive last index in sorted input order.

    Returns:
        list[str]: Selected HDF5 paths.
    """
    hdf5_files = sorted(glob.glob(os.path.join(reference_root, "*.h5")))

    if start is not None and end is not None:
        hdf5_files = hdf5_files[start:end + 1]

    return hdf5_files


def get_reference_data(hdf5_path: str) -> dict:
    """Read frame paths, normalized depth and optional camera extrinsics.

    Args:
        hdf5_path (str): Source HDF5 file path.

    Returns:
        dict: frame_paths (N), target_h/target_w, depth_norm (N, H, W), and transformation (N, 4, 4) or None.
    """
    with h5py.File(hdf5_path, 'r') as f:

        frame_paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['frame_path'][:]]

        depth_data = f['depth'][:]

        transformation = None
        if 'transformation' in f:
            transformation = f['transformation'][:]

        return {
            'frame_paths': frame_paths,
            'target_h': depth_data.shape[1],
            'target_w': depth_data.shape[2],
            'depth_norm': depth_data,
            'transformation': transformation
        }


def load_and_resize_image(image_path: str, target_h: int, target_w: int) -> torch.Tensor:
    """Load an RGB image at the reference depth resolution.

    Args:
        image_path (str): Input image path relative to the release root.
        target_h (int): Target image height in pixels.
        target_w (int): Target image width in pixels.

    Returns:
        torch.Tensor: RGB values with shape (3, target_h, target_w).
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((target_w, target_h), Image.BILINEAR)
    rgb = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1)
    return rgb


def process_single_frame(
    model: UniDepthV2,
    image_path: str,
    target_h: int,
    target_w: int,
    device: torch.device
) -> np.ndarray | None:
    """Infer metric depth for one image.

    Args:
        model (UniDepthV2): Loaded model in evaluation mode.
        image_path (str): Input image path relative to the release root.
        target_h (int): Target image height in pixels.
        target_w (int): Target image width in pixels.
        device (torch.device): Worker device context; model.infer handles image placement.

    Returns:
        np.ndarray or None: Depth in meters with shape (H, W), or None on failure.
    """
    try:
        rgb = load_and_resize_image(image_path, target_h, target_w)

        with torch.no_grad():
            predictions = model.infer(rgb)

        depth = predictions["depth"].squeeze().cpu().numpy()

        if depth is None or depth.size == 0 or np.all(depth <= 0):
            return None

        return depth

    except Exception as e:
        print(f"Warning: Failed to process frame {image_path}: {e}")
        return None


def process_video_frames(
    model: UniDepthV2,
    frame_paths: list,
    target_h: int,
    target_w: int,
    device: torch.device
) -> tuple[list, list]:
    """Infer metric depth for each frame while retaining successful indices.

    Args:
        model (UniDepthV2): Loaded model in evaluation mode.
        frame_paths (list[str]): N image paths relative to the common release root.
        target_h (int): Target image height in pixels.
        target_w (int): Target image width in pixels.
        device (torch.device or str): Device used for inference.

    Returns:
        tuple[list[np.ndarray], list[int]]: Valid (H, W) depths in meters and their original frame indices.
    """
    depths = []
    valid_indices = []

    for i, frame_path in enumerate(frame_paths):
        depth = process_single_frame(model, frame_path, target_h, target_w, device)

        if depth is not None:
            depths.append(depth)
            valid_indices.append(i)

        torch.cuda.empty_cache()

    return depths, valid_indices


def compute_scale_per_frame(depth_real: np.ndarray, depth_norm: np.ndarray) -> float:
    """Estimate metric scale from the median pixel depth ratio.

    Args:
        depth_real (np.ndarray): Metric camera depth with shape (H, W), in meters.
        depth_norm (np.ndarray): Normalized camera depth with shape (H, W).

    Returns:
        float: Metric/normalized depth ratio, or 1.0 when no reference pixels are valid.
    """
    valid_mask = depth_norm > 1e-6

    if not np.any(valid_mask):
        return 1.0

    ratio = depth_real[valid_mask] / depth_norm[valid_mask]
    return np.median(ratio)


def compute_video_scale(
    depths_real: list,
    depths_norm: np.ndarray,
    valid_indices: list
) -> float:
    """Take the median frame scale across successful predictions.

    Args:
        depths_real (list[np.ndarray]): Metric (H, W) depth maps for valid frames.
        depths_norm (np.ndarray): Reference normalized depth with shape (N, H, W).
        valid_indices (list[int]): Indices associating valid depth maps with source frames.

    Returns:
        float: Video-level metric scale, or 1.0 when no predictions are available.
    """
    if len(depths_real) == 0:
        return 1.0

    scales = []

    for i, depth_real in enumerate(depths_real):
        idx = valid_indices[i]
        scale = compute_scale_per_frame(depth_real, depths_norm[idx])
        scales.append(scale)

    return np.median(scales)


def scale_transformation_matrix(transformation: np.ndarray, scale: float) -> np.ndarray:
    """Scale only the translation of each camera extrinsic.

    Args:
        transformation (np.ndarray): World-to-camera extrinsics with shape (N, 4, 4).
        scale (float): Multiplier converting normalized lengths to meters.

    Returns:
        np.ndarray: Copied extrinsics with shape (N, 4, 4) and metric translations.
    """
    scaled_transformation = transformation.copy()
    scaled_transformation[:, :3, 3] *= scale
    return scaled_transformation


def save_to_hdf5(
    output_path: str,
    frame_paths: list,
    transformation_scaled: np.ndarray
) -> None:
    """Write this stage's frame paths and geometry to HDF5.

    Args:
        output_path (str or Path): Destination HDF5 file path.
        frame_paths (list[str]): N image paths relative to the common release root.
        transformation_scaled (np.ndarray or None): Metric world-to-camera extrinsics with shape (N, 4, 4).

    Returns:
        None: Writes HDF5 datasets and frame-count metadata.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('frame_path', data=frame_paths, dtype=dt)

        if transformation_scaled is not None:
            f.create_dataset(
                'transformation',
                data=transformation_scaled.astype(np.float32),
                compression='gzip',
                compression_opts=4
            )

        f.attrs['num_frames'] = len(frame_paths)


def process_single_hdf5(
    reference_path: str,
    output_root: str,
    model: UniDepthV2,
    device: torch.device,
    gpu_id: int = 0,
    min_valid_ratio: float = 0.9
) -> bool:
    """Recover metric scale and save camera extrinsics for one video.

    Args:
        reference_path (str): VGGT HDF5 file containing paths, depth and extrinsics.
        output_root (str or Path): Directory for generated HDF5 files.
        model (UniDepthV2): Loaded model in evaluation mode.
        device (torch.device or str): Device used for inference.
        gpu_id (int): Worker identifier used for logs and progress display.
        min_valid_ratio (float): Minimum fraction of frames with successful depth inference.

    Returns:
        bool: True if output already exists or processing succeeds.
    """
    video_name = os.path.splitext(os.path.basename(reference_path))[0]
    output_path = os.path.join(output_root, f"{video_name}.h5")

    if os.path.exists(output_path):
        return True

    try:

        ref_data = get_reference_data(reference_path)
        frame_paths = ref_data['frame_paths']
        target_h = ref_data['target_h']
        target_w = ref_data['target_w']
        depths_norm = ref_data['depth_norm']
        transformation = ref_data['transformation']

        if transformation is None:
            print(f"[GPU {gpu_id}] Warning: No transformation {reference_path}")
            return False

        missing_frames = [p for p in frame_paths if not os.path.exists(p)]
        if missing_frames:
            print(f"[GPU {gpu_id}] Warning: {len(missing_frames)} frame files missing; skipping {video_name}")
            return False

        depths_real, valid_indices = process_video_frames(
            model, frame_paths, target_h, target_w, device
        )

        valid_ratio = len(valid_indices) / len(frame_paths)
        if valid_ratio < min_valid_ratio:
            print(f"[GPU {gpu_id}] Warning: Valid-frame fraction too low {valid_ratio:.1%}; Skip {video_name}")
            return False

        if len(valid_indices) < len(frame_paths):
            print(f"[GPU {gpu_id}] Info: {video_name} valid frames {len(valid_indices)}/{len(frame_paths)}")

        scale = compute_video_scale(depths_real, depths_norm, valid_indices)

        transformation_scaled = scale_transformation_matrix(transformation, scale)

        save_to_hdf5(output_path, frame_paths, transformation_scaled)

        return True

    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {reference_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the configured processing stage.

    Returns:
        None: Results are written to the configured output directories.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU {args.gpu_id}] Device: {device}")

    print(f"[GPU {args.gpu_id}] Loading model...")
    model = load_model(args.model_path, device)
    print(f"[GPU {args.gpu_id}] Model loaded")

    hdf5_files = get_hdf5_files(args.reference_root, args.start, args.end)
    print(f"[GPU {args.gpu_id}] HDF5 files to process: {len(hdf5_files)}")
    print(f"[GPU {args.gpu_id}] Minimum valid-frame fraction: {args.min_valid_ratio:.0%}")

    os.makedirs(args.output_root, exist_ok=True)

    success_count = 0

    for hdf5_path in tqdm(
        hdf5_files,
        desc=f"[GPU {args.gpu_id}] Progress",
        ncols=100,
        position=0
    ):
        if process_single_hdf5(
            hdf5_path,
            args.output_root,
            model,
            device,
            args.gpu_id,
            args.min_valid_ratio
        ):
            success_count += 1

    print(f"[GPU {args.gpu_id}] Processing complete: {success_count}/{len(hdf5_files)}")

if __name__ == "__main__":
    main()
