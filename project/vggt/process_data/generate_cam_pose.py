"""Extract stitched camera extrinsics and normalized depth from RGB frames."""

import os
import argparse
import glob
import h5py
import numpy as np
import torch
from tqdm import tqdm
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    """Parse command-line options for this processing stage.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(description="VGGT video geometry extraction")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Input root containing video subdirectories")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for HDF5 files")
    parser.add_argument("--model_path", type=str,
                        default="model/VGGT-1B",
                        help="VGGT model directory")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Maximum frames per inference window")
    parser.add_argument("--start", type=int, default=None,
                        help="Inclusive first video index for multi-GPU processing")
    parser.add_argument("--end", type=int, default=None,
                        help="Inclusive last video index for multi-GPU processing")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU identifier for logging only")
    parser.add_argument("--overlap", type=int, default=1,
                        help="Shared frames between windows; only 1 is supported")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch_size must be >= 2")
    if args.overlap != 1:
        parser.error("This implementation only supports --overlap 1")
    return args


def load_model(model_path: str, device: torch.device) -> VGGT:
    """Load the pretrained model and select evaluation mode.

    Args:
        model_path (str): Model checkpoint directory, relative to the release root.
        device (torch.device or str): Device used for inference.

    Returns:
        VGGT: The pretrained model on the requested device.
    """
    model = VGGT.from_pretrained(model_path).to(device)
    model.eval()
    return model


def get_video_dirs(input_root: str, start: int = None, end: int = None) -> list:
    """List video directories in sorted order within an inclusive range.

    Args:
        input_root (str or Path): Root containing one RGB frame directory per video.
        start (int or None): Inclusive first index in sorted input order.
        end (int or None): Inclusive last index in sorted input order.

    Returns:
        list[str]: Selected video directory paths.
    """
    video_dirs = sorted([
        d for d in glob.glob(os.path.join(input_root, "*"))
        if os.path.isdir(d)
    ])

    if start is not None and end is not None:
        video_dirs = video_dirs[start:end + 1]

    return video_dirs


def get_frame_paths(video_dir: str, extensions: tuple = (".jpg", ".png", ".jpeg")) -> list:
    """List image paths in temporal filename order.

    Args:
        video_dir (str): Directory containing temporally sorted RGB frames.
        extensions (tuple[str, ...]): Image suffixes to include, also checked in uppercase.

    Returns:
        list[str]: Sorted frame paths.
    """
    frame_paths = []
    for ext in extensions:
        frame_paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        frame_paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))

    frame_paths = sorted(frame_paths)
    return frame_paths


def extrinsic_to_transformation_matrix(extrinsic: np.ndarray) -> np.ndarray:
    """Extend a world-to-camera extrinsic to homogeneous form.

    Args:
        extrinsic (np.ndarray): World-to-camera [R | t] array with shape (3, 4).

    Returns:
        np.ndarray: Float32 world-to-camera matrix with shape (4, 4).
    """
    transformation = np.eye(4, dtype=np.float32)
    transformation[:3, :] = extrinsic
    return transformation


def estimate_overlap_scale(reference_depth, current_depth):
    """Estimate the new window scale from shared-frame depth.

    Camera translations and depth must both use this multiplier.

    Args:
        reference_depth (np.ndarray): Shared-frame (H, W) depth in the stitched scale.
        current_depth (np.ndarray): Shared-frame (H, W) depth in the new window scale.

    Returns:
        float: Median reference/current depth ratio over finite positive pixels.

    Raises:
        ValueError: Depth shapes differ or no valid positive scale is available.
    """
    if reference_depth.shape != current_depth.shape:
        raise ValueError(
            f"Shared depth shapes differ: {reference_depth.shape} vs {current_depth.shape}"
        )
    valid = (
        np.isfinite(reference_depth) & np.isfinite(current_depth)
        & (reference_depth > 1e-6) & (current_depth > 1e-6)
    )
    if not np.any(valid):
        raise ValueError("No valid shared-frame depth pixels for alignment")
    ratio = (reference_depth[valid].astype(np.float64)
             / current_depth[valid].astype(np.float64))
    scale = float(np.median(ratio))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid overlap scale: {scale}")
    return scale


def process_video_frames(
    model: VGGT,
    frame_paths: list,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 8
) -> tuple:
    """Stitch overlapping windows into one extrinsic reference and scale.

    Consecutive windows share one frame. Depth ratios align scale before
    E_k @ inverse(E_shared) @ E_previous aligns the extrinsic reference.
    The shared frame is retained only once. UniDepth recovers metric scale.

    Args:
        model (VGGT): Loaded model in evaluation mode.
        frame_paths (list[str]): N image paths relative to the common release root.
        device (torch.device or str): Device used for inference.
        dtype (torch.dtype): Autocast precision used for inference.
        batch_size (int): Maximum frames per VGGT window; must be at least two.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: N world-to-camera transforms (4, 4) and camera depths (H, W) in the first window scale.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2")
    num_frames = len(frame_paths)
    if num_frames == 0:
        raise ValueError("No input frames")
    all_transformations = []
    all_depths = []
    batch_start = 0

    while batch_start < num_frames:
        batch_end = min(batch_start + batch_size, num_frames)
        batch_paths = frame_paths[batch_start:batch_end]

        images = load_and_preprocess_images(batch_paths).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                extrinsic, _ = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], images.shape[-2:]
                )
                extrinsic_np = extrinsic[0].float().cpu().numpy()
                depth_np = (predictions["depth"][0, :, :, :, 0]
                            .float().cpu().numpy())

        count = len(batch_paths)
        if extrinsic_np.shape != (count, 3, 4):
            raise ValueError(f"Unexpected extrinsic shape: {extrinsic_np.shape}")
        if depth_np.ndim != 3 or depth_np.shape[0] != count:
            raise ValueError(f"Unexpected depth shape: {depth_np.shape}")
        window_E = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
        window_E[:, :3, :] = extrinsic_np
        window_depth = depth_np.astype(np.float64)
        if not np.isfinite(window_E).all():
            raise ValueError("Window extrinsics contain NaN or Inf")

        keep_from = 0
        if batch_start > 0:
            if len(all_transformations) != batch_start + 1:
                raise RuntimeError("Shared-frame index mismatch")
            reference_E = all_transformations[-1]
            scale = estimate_overlap_scale(all_depths[-1], window_depth[0])
            window_E[:, :3, 3] *= scale
            window_depth *= scale
            # E maps local world -> camera. Match both shared-frame extrinsics.
            connection = np.linalg.inv(window_E[0]) @ reference_E
            window_E = window_E @ connection
            if not np.isfinite(window_E).all():
                raise ValueError("Aligned extrinsics contain NaN or Inf")
            keep_from = 1  # Shared frame has already been saved.

        for index in range(keep_from, count):
            all_transformations.append(window_E[index].copy())
            all_depths.append(window_depth[index].copy())

        del images, predictions, extrinsic
        torch.cuda.empty_cache()
        if batch_end == num_frames:
            break
        batch_start = batch_end - 1

    if len(all_transformations) != num_frames or len(all_depths) != num_frames:
        raise RuntimeError("Output frame count does not match input")
    return all_transformations, all_depths


def save_to_hdf5(
    output_path: str,
    frame_paths: list,
    transformations: list,
    depths: list
) -> None:
    """Write this stage's frame paths and geometry to HDF5.

    Args:
        output_path (str or Path): Destination HDF5 file path.
        frame_paths (list[str]): N image paths relative to the common release root.
        transformations (list[np.ndarray]): N stitched world-to-camera matrices of shape (4, 4).
        depths (list[np.ndarray]): N camera depth maps of shape (H, W) in the stitched scale.

    Returns:
        None: Writes HDF5 datasets and frame-count metadata.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    transformations_array = np.stack(transformations, axis=0).astype(np.float32)
    depths_array = np.stack(depths, axis=0).astype(np.float32)

    relative_paths = [os.path.relpath(p) for p in frame_paths]

    with h5py.File(output_path, 'w') as f:

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('frame_path', data=relative_paths, dtype=dt)

        f.create_dataset(
            'transformation',
            data=transformations_array,
            compression='gzip',
            compression_opts=4
        )

        f.create_dataset(
            'depth',
            data=depths_array,
            compression='gzip',
            compression_opts=4
        )

        f.attrs['num_frames'] = len(frame_paths)
        f.attrs['depth_shape'] = depths_array.shape[1:]


def process_single_video(
    video_dir: str,
    output_root: str,
    model: VGGT,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int
) -> bool:
    """Extract and save stitched camera geometry for one video.

    Args:
        video_dir (str): Directory containing temporally sorted RGB frames.
        output_root (str or Path): Directory for generated HDF5 files.
        model (VGGT): Loaded model in evaluation mode.
        device (torch.device or str): Device used for inference.
        dtype (torch.dtype): Autocast precision used for inference.
        batch_size (int): Maximum frames per VGGT window; must be at least two.

    Returns:
        bool: True if output already exists or processing succeeds.
    """
    video_name = os.path.basename(video_dir)
    output_path = os.path.join(output_root, f"{video_name}.h5")

    if os.path.exists(output_path):
        return True

    frame_paths = get_frame_paths(video_dir)
    if len(frame_paths) == 0:
        print(f"Warning: {video_dir} contains no image files")
        return False

    try:

        transformations, depths = process_video_frames(
            model, frame_paths, device, dtype, batch_size
        )

        save_to_hdf5(output_path, frame_paths, transformations, depths)
        return True

    except Exception as e:
        print(f"Error: Processing {video_dir} raised an exception: {e}")
        return False


def main():
    """Run the configured processing stage.

    Returns:
        None: Results are written to the configured output directories.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"[GPU {args.gpu_id}] Device: {device}, Precision: {dtype}")

    print(f"[GPU {args.gpu_id}] Loading model...")
    model = load_model(args.model_path, device)
    print(f"[GPU {args.gpu_id}] Model loaded")

    video_dirs = get_video_dirs(args.input_root, args.start, args.end)
    print(f"[GPU {args.gpu_id}] Videos to process: {len(video_dirs)}")

    os.makedirs(args.output_root, exist_ok=True)

    success_count = 0
    for video_dir in tqdm(video_dirs, desc=f"[GPU {args.gpu_id}] Progress"):
        if process_single_video(
            video_dir, args.output_root, model, device, dtype, args.batch_size
        ):
            success_count += 1

    print(f"[GPU {args.gpu_id}] Processing complete: {success_count}/{len(video_dirs)}")

if __name__ == "__main__":
    main()
