import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from notebook.utils import setup_sam_3d_body


def create_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Combine a rotation and translation into a homogeneous transform.

    Args:
        rotation (np.ndarray): Rotation matrix with shape (3, 3).
        translation (np.ndarray): Translation vector with shape (3,), in meters.

    Returns:
        np.ndarray: Transform with shape (4, 4).
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def extract_wrist_poses_batch(outputs: List[dict]) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool]]:
    """Extract wrist-to-camera transforms from the first detected person.

    Args:
        outputs (list): Per-frame prediction lists; wrist poses are extracted from the first detected person.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[bool]]: Left/right transforms of shape (4, 4) and per-frame validity flags. Use entries selected by these flags.
    """
    left_wrist_T_list = []
    right_wrist_T_list = []
    valid_mask = []

    left_wrist_idx = 78
    right_wrist_idx = 42

    for output in outputs:
        if output is None or len(output) == 0:
            left_wrist_T_list.append(np.eye(4))
            right_wrist_T_list.append(np.eye(4))
            valid_mask.append(False)
            continue

        try:
            pred_coords = output[0]['pred_joint_coords']
            pred_rots = output[0]['pred_global_rots']
            camera_translation = output[0]['pred_cam_t']

            left_coords = pred_coords[left_wrist_idx] + camera_translation
            left_rot = pred_rots[left_wrist_idx]
            left_T = create_transformation_matrix(left_rot, left_coords)

            right_coords = pred_coords[right_wrist_idx] + camera_translation
            right_rot = pred_rots[right_wrist_idx]
            right_T = create_transformation_matrix(right_rot, right_coords)

            left_wrist_T_list.append(left_T)
            right_wrist_T_list.append(right_T)
            valid_mask.append(True)

        except Exception:
            left_wrist_T_list.append(np.eye(4))
            right_wrist_T_list.append(np.eye(4))
            valid_mask.append(False)

    return left_wrist_T_list, right_wrist_T_list, valid_mask


def get_sorted_frame_paths(video_folder: Path, frame_interval: int = 3) -> List[Path]:
    """Sample sorted video frames at the requested stride.

    Args:
        video_folder (Path): Directory containing temporally sorted RGB frames.
        frame_interval (int): Stride through sorted source frames.

    Returns:
        list[Path]: Sampled image paths in temporal order.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    all_frames = sorted([
        f for f in video_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    sampled_frames = all_frames[::frame_interval]
    return sampled_frames


def get_video_subfolders(root_folder: Path) -> List[Path]:
    """List immediate video subdirectories in sorted order.

    Args:
        root_folder (Path): Directory containing video subdirectories.

    Returns:
        list[Path]: Video directories.
    """
    return sorted([f for f in root_folder.iterdir() if f.is_dir()])


def process_video_folder(
    estimator,
    video_folder: Path,
    output_path: Path,
    frame_interval: int = 3
) -> Tuple[bool, int, int]:
    """Infer wrist poses and save only successfully detected frames.

    Args:
        estimator (SAM3DBodyEstimator): Configured body estimator.
        video_folder (Path): Directory containing temporally sorted RGB frames.
        output_path (str or Path): Destination HDF5 file path.
        frame_interval (int): Stride through sorted source frames.

    Returns:
        tuple[bool, int, int]: Success, retained frame count and sampled frame count.
    """
    frame_paths = get_sorted_frame_paths(video_folder, frame_interval)

    if len(frame_paths) == 0:
        return False, 0, 0

    all_outputs = []
    for path in frame_paths:
        try:
            output = estimator.process_one_image(str(path))
            all_outputs.append(output)
        except Exception:
            all_outputs.append(None)

    left_T_list, right_T_list, valid_mask = extract_wrist_poses_batch(all_outputs)

    valid_indices = []
    valid_frame_paths = []
    valid_left_T = []
    valid_right_T = []

    for idx, (is_valid, frame_path, left_T, right_T) in enumerate(
        zip(valid_mask, frame_paths, left_T_list, right_T_list)
    ):
        if is_valid:
            valid_indices.append(idx)
            valid_frame_paths.append(os.path.relpath(frame_path))
            valid_left_T.append(left_T)
            valid_right_T.append(right_T)

    if len(valid_indices) == 0:
        return False, 0, len(frame_paths)

    save_to_hdf5(output_path, valid_indices, valid_frame_paths, valid_left_T, valid_right_T)

    return True, len(valid_indices), len(frame_paths)


def save_to_hdf5(
    output_path: Path,
    indices: List[int],
    frame_paths: List[str],
    left_wrist_T: List[np.ndarray],
    right_wrist_T: List[np.ndarray]
) -> None:
    """Write this stage's frame paths and geometry to HDF5.

    Args:
        output_path (str or Path): Destination HDF5 file path.
        indices (list[int]): Original indices of retained frames, used for frame count.
        frame_paths (list[str]): N image paths relative to the common release root.
        left_wrist_T (list[np.ndarray]): N wrist-to-camera matrices of shape (4, 4), in meters.
        right_wrist_T (list[np.ndarray]): N wrist-to-camera matrices of shape (4, 4), in meters.

    Returns:
        None: Writes HDF5 datasets and frame-count metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # f.create_dataset('index', data=np.array(indices, dtype=np.int32))

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('frame_path', data=frame_paths, dtype=dt)

        f.create_dataset('left_wrist_T', data=np.array(left_wrist_T, dtype=np.float32))
        f.create_dataset('right_wrist_T', data=np.array(right_wrist_T, dtype=np.float32))

        f.attrs['num_frames'] = len(indices)


def process_videos_range(
    model_path: str,
    input_root: Path,
    output_root: Path,
    start_idx: int,
    end_idx: int,
    gpu_id: int,
    frame_interval: int = 3
) -> None:
    """Process an inclusive video range on one worker.

    Args:
        model_path (str): Model checkpoint directory, relative to the release root.
        input_root (str or Path): Root containing one RGB frame directory per video.
        output_root (str or Path): Directory for generated HDF5 files.
        start_idx (int): Inclusive first video index.
        end_idx (int): Inclusive last video index.
        gpu_id (int): Worker identifier used for logs and progress display.
        frame_interval (int): Stride through sorted source frames.

    Returns:
        None: New wrist HDF5 files are written; existing outputs are skipped.
    """

    video_folders = get_video_subfolders(input_root)
    total_videos = len(video_folders)

    if total_videos == 0:
        print(f"[GPU {gpu_id}] Error: No video subdirectories found - {input_root}")
        return

    start_idx = max(0, start_idx)
    end_idx = min(end_idx, total_videos - 1)

    if start_idx > end_idx:
        print(f"[GPU {gpu_id}] Error: Invalid index range [{start_idx}, {end_idx}]")
        return

    selected_folders = video_folders[start_idx:end_idx + 1]

    pending = []
    skipped = 0
    for vf in selected_folders:
        output_path = output_root / f"{vf.name}.h5"
        if output_path.exists():
            skipped += 1
        else:
            pending.append((vf, output_path))

    print(f"[GPU {gpu_id}] Range: [{start_idx}, {end_idx}] | Pending: {len(pending)} | Skipped: {skipped}")

    if len(pending) == 0:
        print(f"[GPU {gpu_id}] All outputs already exist; nothing to process!")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[GPU {gpu_id}] Initializing model...")
    estimator = setup_sam_3d_body(model_path)
    print(f"[GPU {gpu_id}] Model loaded!")

    success_count = 0
    pbar = tqdm(
        pending,
        desc=f"GPU {gpu_id}",
        unit="video",
        position=gpu_id,
        leave=True,
        ncols=100
    )

    for video_folder, output_path in pbar:

        pbar.set_postfix_str(f"{video_folder.name[:20]}...")

        success, valid_frames, total_frames = process_video_folder(
            estimator, video_folder, output_path, frame_interval
        )

        if success:
            success_count += 1

    pbar.close()
    print(f"[GPU {gpu_id}] Done! Success: {success_count}/{len(pending)}")


def main():
    """Run the configured processing stage.

    Returns:
        None: Results are written to the configured output directories.
    """
    parser = argparse.ArgumentParser(description="SAM 3D Body video processing")

    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Inclusive first video index, starting from zero"
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Inclusive last video index"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=True,
        help="GPU identifier for progress display"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/sam-3d-body-vith",
        help="model directory"
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/ego4d_fho/v2/segments",
        help="Root directory containing video frames"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/ego4d_fho/v2/hdf5s",
        help="Output root for HDF5 files"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Sampling stride; retain one in every frame_interval frames"
    )

    args = parser.parse_args()

    process_videos_range(
        model_path=args.model_path,
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        start_idx=args.start,
        end_idx=args.end,
        gpu_id=args.gpu_id,
        frame_interval=args.frame_interval
    )

if __name__ == "__main__":
    main()
