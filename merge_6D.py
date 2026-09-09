#!/usr/bin/env python3
"""Merge camera and wrist geometry using robot axes and sequential rotation columns."""

import os
import sys
import h5py
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class Logger:
    """Mirror terminal messages to a timestamped log file."""

    def __init__(self, log_dir: str):
        """Open a timestamped log while retaining the terminal stream.

        Args:
            log_dir (str): Directory for timestamped log files.

        Returns:
            None: Initializes the output streams.
        """
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"merge_log_{timestamp}.txt"
        self.log_path = os.path.join(log_dir, log_filename)

        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.terminal = sys.stdout

    def write(self, message: str):
        """Forward text to the terminal and log file.

        Args:
            message (str): Text forwarded to both terminal and log file.

        Returns:
            None: Flushes the file after writing.
        """
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        """Flush both output streams.

        Returns:
            None: Buffered text is forwarded to the streams.
        """
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        """Close the log file.

        Returns:
            None: Releases the file handle.
        """
        self.log_file.close()


@dataclass
class Config:
    """Configure merge paths, rotation tolerances and filtering."""

    sam3d_root: str = "data/ego4d_fho/v2/hdf5s/sam-3d-body"
    unidepth_root: str = "data/ego4d_fho/v2/hdf5s/unidepth_overlap1"

    output_root: str = "data/ego4d_fho/v2/hdf5s/merged_overlap1"

    log_dir: str = "logs/merge"

    check_frame_existence: bool = True
    check_rotation_matrix: bool = True
    skip_existing: bool = True

    convert_coordinate_system: bool = True

    ortho_tolerance: float = 1e-4
    det_tolerance: float = 1e-4

    verbose: bool = False


def get_opencv_to_robot_rotation() -> np.ndarray:
    """Map OpenCV right/down/forward axes to robot forward/left/up axes.

    Returns:
        np.ndarray: Axis-conversion rotation with shape (3, 3).
    """
    R = np.array([
        [0,  0,  1],
        [-1, 0,  0],
        [0, -1,  0]
    ], dtype=np.float64)
    return R


def transform_T_to_new_coordinate(T_opencv: np.ndarray) -> np.ndarray:
    """Change both transform bases from OpenCV axes to robot axes.

    Args:
        T_opencv (np.ndarray): Transform with shape (4, 4), expressed using OpenCV axes.

    Returns:
        np.ndarray: Shape (4, 4), computed as C @ T_opencv @ inverse(C).
    """
    R_conv = get_opencv_to_robot_rotation()

    T_conv = np.eye(4, dtype=np.float64)
    T_conv[:3, :3] = R_conv

    T_conv_inv = np.linalg.inv(T_conv)
    # Change both bases so translations and local wrist axes use robot axes.
    T_new = T_conv @ T_opencv @ T_conv_inv

    return T_new


def transform_batch_T_to_new_coordinate(T_batch: np.ndarray) -> np.ndarray:
    """Change transform bases from OpenCV axes to robot axes.

    Args:
        T_batch (np.ndarray): Transforms with shape (N, 4, 4), or a single (4, 4) matrix.

    Returns:
        np.ndarray: Converted transforms, preserving the input shape.
    """
    if T_batch.ndim == 2:
        return transform_T_to_new_coordinate(T_batch)

    R_conv = get_opencv_to_robot_rotation()

    T_conv = np.eye(4, dtype=np.float64)
    T_conv[:3, :3] = R_conv
    T_conv_inv = np.linalg.inv(T_conv)

    T_new_batch = T_conv @ T_batch @ T_conv_inv

    return T_new_batch


def T_to_pos_rot6d(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode translation and the first two rotation columns in sequence.

    Args:
        T (np.ndarray): Homogeneous transform with shape (4, 4).

    Returns:
        tuple[np.ndarray, np.ndarray]: Position (3,) and rotation (6,) ordered [r11, r21, r31, r12, r22, r32].
    """

    position = T[:3, 3].copy()

    R = T[:3, :3]
    # The input decoder expects complete column vectors in sequence.
    rotation_6d = np.concatenate([R[:, 0], R[:, 1]])

    return position, rotation_6d


def batch_T_to_pos_rot6d(T_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a transform batch using sequential rotation columns.

    Args:
        T_batch (np.ndarray): Transforms with shape (N, 4, 4), or a single (4, 4) matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: Positions (N, 3) and rotations (N, 6); a single transform yields N=1.
    """
    if T_batch.ndim == 2:
        pos, rot6d = T_to_pos_rot6d(T_batch)
        return pos.reshape(1, 3), rot6d.reshape(1, 6)

    N = T_batch.shape[0]

    positions = T_batch[:, :3, 3].copy()  # (N, 3)

    R_batch = T_batch[:, :3, :3]  # (N, 3, 3)
    col1 = R_batch[:, :, 0]  # (N, 3)
    col2 = R_batch[:, :, 1]  # (N, 3)
    rotations_6d = np.concatenate([col1, col2], axis=1)  # (N, 6)

    return positions, rotations_6d


def rot6d_to_rotation_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Recover a rotation from sequential columns using Gram-Schmidt.

    Args:
        rot6d (np.ndarray): Shape (6,); sequential columns [r11, r21, r31, r12, r22, r32].

    Returns:
        np.ndarray: Orthonormal rotation with shape (3, 3).
    """
    col1 = rot6d[:3]
    col2 = rot6d[3:6]

    col1 = col1 / np.linalg.norm(col1)

    col2 = col2 - np.dot(col2, col1) * col1
    col2 = col2 / np.linalg.norm(col2)

    col3 = np.cross(col1, col2)

    R = np.column_stack([col1, col2, col3])
    return R


def load_hdf5_data(hdf5_path: str, keys: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """Read required datasets from an HDF5 file.

    Args:
        hdf5_path (str): Source HDF5 file path.
        keys (list[str]): Required HDF5 dataset names.

    Returns:
        dict[str, np.ndarray] or None: Dataset arrays, or None if the file or a key is missing.
    """
    if not os.path.exists(hdf5_path):
        return None

    data = {}
    with h5py.File(hdf5_path, 'r') as f:
        for key in keys:
            if key in f:
                data[key] = f[key][:]
            else:
                print(f"Warning: {hdf5_path} is missing {key}")
                return None
    return data


def decode_frame_paths(encoded_paths: np.ndarray) -> List[str]:
    """Decode stored frame paths without changing their path base.

    Args:
        encoded_paths (np.ndarray): One-dimensional array of byte or string frame paths.

    Returns:
        list[str]: Decoded paths in stored order.
    """
    paths = []
    for p in encoded_paths:
        if isinstance(p, bytes):
            paths.append(p.decode('utf-8'))
        elif isinstance(p, np.bytes_):
            paths.append(p.decode('utf-8'))
        else:
            paths.append(str(p))
    return paths


def validate_transformation_matrix(
    T: np.ndarray,
    ortho_tol: float,
    det_tol: float
) -> Tuple[bool, str]:
    """Check transform shape, finite values and rotation validity.

    Args:
        T (np.ndarray): Homogeneous transform with shape (4, 4).
        ortho_tol (float): Maximum absolute error in R @ R.T relative to identity.
        det_tol (float): Maximum absolute difference between det(R) and one.

    Returns:
        tuple[bool, str]: Whether the shape, finite-value and rotation checks pass, and a diagnostic message.
    """
    if T.shape != (4, 4):
        return False, f"Invalid matrix shape: {T.shape}"

    if not np.isfinite(T).all():
        return False, "Matrix contains NaN or Inf"

    R = T[:3, :3]

    RRT = R @ R.T
    identity = np.eye(3)
    ortho_error = np.abs(RRT - identity).max()
    if ortho_error > ortho_tol:
        return False, f"Orthogonality error too large: {ortho_error:.6f}"

    det = np.linalg.det(R)
    det_error = np.abs(det - 1.0)
    if det_error > det_tol:
        return False, f"Determinant error too large: det={det:.6f}"

    return True, ""


def validate_frame_transformations(
    T_lwrist: np.ndarray,
    T_rwrist: np.ndarray,
    T_cam: np.ndarray,
    config: Config
) -> Tuple[bool, List[str]]:
    """Validate both wrist matrices and the camera matrix for one frame.

    Args:
        T_lwrist (np.ndarray): Left wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_rwrist (np.ndarray): Right wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_cam (np.ndarray): Camera transform, (4, 4) per frame or (N, 4, 4) for a batch.
        config (Config): Paths, filtering options and rotation tolerances.

    Returns:
        tuple[bool, list[str]]: Whether all matrices pass, and any diagnostics.
    """
    errors = []

    valid, msg = validate_transformation_matrix(
        T_lwrist, config.ortho_tolerance, config.det_tolerance
    )
    if not valid:
        errors.append(f"left_wrist_T: {msg}")

    valid, msg = validate_transformation_matrix(
        T_rwrist, config.ortho_tolerance, config.det_tolerance
    )
    if not valid:
        errors.append(f"right_wrist_T: {msg}")

    valid, msg = validate_transformation_matrix(
        T_cam, config.ortho_tolerance, config.det_tolerance
    )
    if not valid:
        errors.append(f"transformation: {msg}")

    return len(errors) == 0, errors


def check_frame_exists(frame_path: str) -> bool:
    """Check whether the frame path points to a file.

    Args:
        frame_path (str): Input frame path relative to the release root.

    Returns:
        bool: True when the frame file exists.
    """
    return os.path.isfile(frame_path)


def check_output_exists(output_path: str) -> bool:
    """Check HDF5 readability and the required dataset names.

    Args:
        output_path (str or Path): Destination HDF5 file path.

    Returns:
        bool: True if all checked datasets exist and frame_path is nonempty.
    """
    if not os.path.exists(output_path):
        return False

    try:
        with h5py.File(output_path, 'r') as f:
            required_keys = [
                'frame_path',
                'lwrist_pos_cami', 'lwrist_rot6d_cami',
                'rwrist_pos_cami', 'rwrist_rot6d_cami',
                'cam_pos_cam1', 'cam_rot6d_cam1'
            ]
            for key in required_keys:
                if key not in f:
                    return False

            if len(f['frame_path']) == 0:
                return False
        return True
    except Exception:
        return False


def match_frames_by_path(
    paths_sam3d: List[str],
    paths_unidepth: List[str],
    data_sam3d: Dict[str, np.ndarray],
    data_unidepth: Dict[str, np.ndarray]
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Join wrist and camera data using identical stored frame paths.

    Args:
        paths_sam3d (list[str]): SAM frame paths in their stored order.
        paths_unidepth (list[str]): UniDepth frame paths in their stored order.
        data_sam3d (dict[str, np.ndarray]): SAM datasets including left and right wrist matrices.
        data_unidepth (dict[str, np.ndarray]): UniDepth datasets including camera extrinsics.

    Returns:
        tuple: Matched paths, left/right/camera arrays (N, 4, 4), and matching counts. Empty matches produce empty arrays.
    """

    unidepth_path_to_idx = {path: idx for idx, path in enumerate(paths_unidepth)}

    matched_paths = []
    matched_lwrist = []
    matched_rwrist = []
    matched_cam = []

    stats = {
        'sam3d_total': len(paths_sam3d),
        'unidepth_total': len(paths_unidepth),
        'matched': 0,
        'sam3d_only': 0,
    }

    for sam3d_idx, path in enumerate(paths_sam3d):
        if path in unidepth_path_to_idx:
            unidepth_idx = unidepth_path_to_idx[path]
            matched_paths.append(path)
            matched_lwrist.append(data_sam3d['left_wrist_T'][sam3d_idx])
            matched_rwrist.append(data_sam3d['right_wrist_T'][sam3d_idx])
            matched_cam.append(data_unidepth['transformation'][unidepth_idx])
            stats['matched'] += 1
        else:
            stats['sam3d_only'] += 1

    if len(matched_paths) > 0:
        matched_lwrist = np.stack(matched_lwrist, axis=0)
        matched_rwrist = np.stack(matched_rwrist, axis=0)
        matched_cam = np.stack(matched_cam, axis=0)
    else:
        matched_lwrist = np.array([])
        matched_rwrist = np.array([])
        matched_cam = np.array([])

    return matched_paths, matched_lwrist, matched_rwrist, matched_cam, stats


def filter_existing_frames_with_data(
    frame_paths: List[str],
    T_lwrist: np.ndarray,
    T_rwrist: np.ndarray,
    T_cam: np.ndarray
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, int]:
    """Retain rows whose image files exist.

    Args:
        frame_paths (list[str]): N image paths relative to the common release root.
        T_lwrist (np.ndarray): Left wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_rwrist (np.ndarray): Right wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_cam (np.ndarray): Camera transform, (4, 4) per frame or (N, 4, 4) for a batch.

    Returns:
        tuple: Paths, left/right/camera arrays (N, 4, 4), and the number removed. Empty results use empty arrays.
    """
    existing_paths = []
    existing_lwrist = []
    existing_rwrist = []
    existing_cam = []
    filtered_count = 0

    for i, path in enumerate(frame_paths):
        if check_frame_exists(path):
            existing_paths.append(path)
            existing_lwrist.append(T_lwrist[i])
            existing_rwrist.append(T_rwrist[i])
            existing_cam.append(T_cam[i])
        else:
            filtered_count += 1

    if len(existing_paths) > 0:
        existing_lwrist = np.stack(existing_lwrist, axis=0)
        existing_rwrist = np.stack(existing_rwrist, axis=0)
        existing_cam = np.stack(existing_cam, axis=0)
    else:
        existing_lwrist = np.array([])
        existing_rwrist = np.array([])
        existing_cam = np.array([])

    return existing_paths, existing_lwrist, existing_rwrist, existing_cam, filtered_count


def filter_valid_rotations_with_data(
    frame_paths: List[str],
    T_lwrist: np.ndarray,
    T_rwrist: np.ndarray,
    T_cam: np.ndarray,
    config: Config,
    stage_name: str = ""
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Retain rows for which all three rotations pass validation.

    Args:
        frame_paths (list[str]): N image paths relative to the common release root.
        T_lwrist (np.ndarray): Left wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_rwrist (np.ndarray): Right wrist transform, (4, 4) per frame or (N, 4, 4) for a batch.
        T_cam (np.ndarray): Camera transform, (4, 4) per frame or (N, 4, 4) for a batch.
        config (Config): Paths, filtering options and rotation tolerances.
        stage_name (str): Optional stage label for validation messages.

    Returns:
        tuple: Paths, left/right/camera arrays (N, 4, 4), and per-field error counts. Empty results use empty arrays.
    """
    valid_paths = []
    valid_lwrist = []
    valid_rwrist = []
    valid_cam = []

    error_stats = {
        'left_wrist_T': 0,
        'right_wrist_T': 0,
        'transformation': 0
    }

    for i, path in enumerate(frame_paths):
        is_valid, errors = validate_frame_transformations(
            T_lwrist[i], T_rwrist[i], T_cam[i], config
        )

        if is_valid:
            valid_paths.append(path)
            valid_lwrist.append(T_lwrist[i])
            valid_rwrist.append(T_rwrist[i])
            valid_cam.append(T_cam[i])
        else:
            for err in errors:
                if 'left_wrist_T' in err:
                    error_stats['left_wrist_T'] += 1
                elif 'right_wrist_T' in err:
                    error_stats['right_wrist_T'] += 1
                elif 'transformation' in err:
                    error_stats['transformation'] += 1

            if config.verbose:
                stage_prefix = f"[{stage_name}] " if stage_name else ""
                print(f"  {stage_prefix}Frame {path} invalid rotation matrix: {errors}")

    if len(valid_paths) > 0:
        valid_lwrist = np.stack(valid_lwrist, axis=0)
        valid_rwrist = np.stack(valid_rwrist, axis=0)
        valid_cam = np.stack(valid_cam, axis=0)
    else:
        valid_lwrist = np.array([])
        valid_rwrist = np.array([])
        valid_cam = np.array([])

    return valid_paths, valid_lwrist, valid_rwrist, valid_cam, error_stats


def merge_single_hdf5(
    sam3d_path: str,
    unidepth_path: str,
    output_path: str,
    config: Config
) -> Dict:
    """Match, invert camera extrinsics, validate and encode one HDF5 pair.

    Args:
        sam3d_path (str): SAM wrist-pose HDF5 file path.
        unidepth_path (str): Metric camera-extrinsic HDF5 file path.
        output_path (str or Path): Destination HDF5 file path.
        config (Config): Paths, filtering options and rotation tolerances.

    Returns:
        dict: Frame counts, skip status and validation error counts.
    """
    stats = {
        'sam3d_frames': 0,
        'unidepth_frames': 0,
        'matched_frames': 0,
        'existing_frames': 0,
        'valid_rotation_frames': 0,
        'post_convert_valid_frames': 0,
        'final_frames': 0,
        'skipped': False,
        'skipped_existing': False,
        'rotation_errors': {
            'left_wrist_T': 0,
            'right_wrist_T': 0,
            'transformation': 0
        },
        'post_convert_rotation_errors': {
            'left_wrist_T': 0,
            'right_wrist_T': 0,
            'transformation': 0
        }
    }

    if config.skip_existing and check_output_exists(output_path):
        stats['skipped'] = True
        stats['skipped_existing'] = True
        return stats

    data_sam3d = load_hdf5_data(
        sam3d_path,
        ['frame_path', 'left_wrist_T', 'right_wrist_T']
    )
    if data_sam3d is None:
        print(f"Skip: Unable to load {sam3d_path}")
        stats['skipped'] = True
        return stats

    data_unidepth = load_hdf5_data(
        unidepth_path,
        ['frame_path', 'transformation']
    )
    if data_unidepth is None:
        print(f"Skip: Unable to load {unidepth_path}")
        stats['skipped'] = True
        return stats

    paths_sam3d = decode_frame_paths(data_sam3d['frame_path'])
    paths_unidepth = decode_frame_paths(data_unidepth['frame_path'])

    stats['sam3d_frames'] = len(paths_sam3d)
    stats['unidepth_frames'] = len(paths_unidepth)

    matched_paths, matched_lwrist, matched_rwrist, matched_cam, match_stats = match_frames_by_path(
        paths_sam3d, paths_unidepth, data_sam3d, data_unidepth
    )
    stats['matched_frames'] = match_stats['matched']

    if config.verbose and match_stats['sam3d_only'] > 0:
        print(f"  {match_stats['sam3d_only']} frames exist only in SAM")

    if len(matched_paths) == 0:
        print(f"Skip: No matching frames")
        stats['skipped'] = True
        return stats

    current_paths = matched_paths
    current_lwrist = matched_lwrist
    current_rwrist = matched_rwrist
    # Metric world-to-camera extrinsics -> camera-to-world poses.
    # np.linalg.inv applies independently to each (4, 4) matrix.
    current_cam = np.linalg.inv(matched_cam)

    if config.check_frame_existence:
        current_paths, current_lwrist, current_rwrist, current_cam, filtered = filter_existing_frames_with_data(
            current_paths, current_lwrist, current_rwrist, current_cam
        )
        if config.verbose and filtered > 0:
            print(f"  {filtered} frame files do not exist")

    stats['existing_frames'] = len(current_paths)

    if len(current_paths) == 0:
        print(f"Skip: No existing frame files")
        stats['skipped'] = True
        return stats

    if config.check_rotation_matrix:
        current_paths, current_lwrist, current_rwrist, current_cam, rotation_errors = filter_valid_rotations_with_data(
            current_paths, current_lwrist, current_rwrist, current_cam, config, stage_name="before axis conversion"
        )
        stats['rotation_errors'] = rotation_errors

    stats['valid_rotation_frames'] = len(current_paths)

    if len(current_paths) == 0:
        print(f"Skip: No valid frames before axis conversion")
        stats['skipped'] = True
        return stats

    if config.convert_coordinate_system:
        current_lwrist = transform_batch_T_to_new_coordinate(current_lwrist)
        current_rwrist = transform_batch_T_to_new_coordinate(current_rwrist)
        current_cam = transform_batch_T_to_new_coordinate(current_cam)

    if config.check_rotation_matrix:
        current_paths, current_lwrist, current_rwrist, current_cam, post_convert_errors = filter_valid_rotations_with_data(
            current_paths, current_lwrist, current_rwrist, current_cam, config, stage_name="after axis conversion"
        )
        stats['post_convert_rotation_errors'] = post_convert_errors

        if config.verbose:
            total_post_errors = sum(post_convert_errors.values())
            if total_post_errors > 0:
                print(f"  After axis conversion, found {total_post_errors} invalid rotation matrices")

    stats['post_convert_valid_frames'] = len(current_paths)
    stats['final_frames'] = len(current_paths)

    if len(current_paths) == 0:
        print(f"Skip: No valid frames after axis conversion")
        stats['skipped'] = True
        return stats

    lwrist_pos, lwrist_rot6d = batch_T_to_pos_rot6d(current_lwrist)
    rwrist_pos, rwrist_rot6d = batch_T_to_pos_rot6d(current_rwrist)
    cam_pos, cam_rot6d = batch_T_to_pos_rot6d(current_cam)

    write_merged_hdf5(
        output_path,
        current_paths,
        lwrist_pos, lwrist_rot6d,
        rwrist_pos, rwrist_rot6d,
        cam_pos, cam_rot6d
    )

    return stats


def write_merged_hdf5(
    output_path: str,
    frame_paths: List[str],
    lwrist_pos: np.ndarray,
    lwrist_rot6d: np.ndarray,
    rwrist_pos: np.ndarray,
    rwrist_rot6d: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot6d: np.ndarray
) -> None:
    """Write merged positions and sequential-column rotations.

    Args:
        output_path (str or Path): Destination HDF5 file path.
        frame_paths (list[str]): N image paths relative to the common release root.
        lwrist_pos (np.ndarray): Left wrist in current-camera coordinates, shape (N, 3), in meters.
        lwrist_rot6d (np.ndarray): Left wrist in current-camera coordinates, shape (N, 6), sequential first two columns.
        rwrist_pos (np.ndarray): Right wrist in current-camera coordinates, shape (N, 3), in meters.
        rwrist_rot6d (np.ndarray): Right wrist in current-camera coordinates, shape (N, 6), sequential first two columns.
        cam_pos (np.ndarray): Camera in the common reference, shape (N, 3), in meters.
        cam_rot6d (np.ndarray): Camera in the common reference, shape (N, 6), sequential first two columns.

    Returns:
        None: Writes the merged HDF5 datasets and coordinate metadata.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    encoded_paths = np.array([p.encode('utf-8') for p in frame_paths], dtype='S')

    with h5py.File(output_path, 'w') as f:

        f.create_dataset('frame_path', data=encoded_paths)

        f.create_dataset('lwrist_pos_cami', data=lwrist_pos.astype(np.float32))
        f.create_dataset('lwrist_rot6d_cami', data=lwrist_rot6d.astype(np.float32))

        f.create_dataset('rwrist_pos_cami', data=rwrist_pos.astype(np.float32))
        f.create_dataset('rwrist_rot6d_cami', data=rwrist_rot6d.astype(np.float32))

        f.create_dataset('cam_pos_cam1', data=cam_pos.astype(np.float32))
        f.create_dataset('cam_rot6d_cam1', data=cam_rot6d.astype(np.float32))

        f.attrs['description'] = 'Merged transformation data with 6D rotation representation'
        f.attrs['coordinate_system'] = 'Robot: +X front, +Y left, +Z up'
        f.attrs['rotation_6d_format'] = 'First two columns of rotation matrix: [r11, r21, r31, r12, r22, r32]'


def get_hdf5_files(directory: str) -> List[str]:
    """List sorted HDF5 filenames in a directory.

    Args:
        directory (str): Directory searched for .h5 and .hdf5 files.

    Returns:
        list[str]: Filenames without directory prefixes.
    """
    files = []
    for f in os.listdir(directory):
        if f.endswith('.h5') or f.endswith('.hdf5'):
            files.append(f)
    return sorted(files)


def process_all_hdf5(config: Config) -> None:
    """Merge matching HDF5 filenames from the two input directories.

    Args:
        config (Config): Paths, filtering options and rotation tolerances.

    Returns:
        None: Writes merged files and prints aggregate statistics.
    """

    files_sam3d = set(get_hdf5_files(config.sam3d_root))
    files_unidepth = set(get_hdf5_files(config.unidepth_root))

    common_files = sorted(files_sam3d & files_unidepth)
    only_in_sam3d = files_sam3d - files_unidepth
    only_in_unidepth = files_unidepth - files_sam3d

    print(f"sam-3d-body file count: {len(files_sam3d)}")
    print(f"unidepth file count: {len(files_unidepth)}")
    print(f"Common filenames: {len(common_files)}")

    if only_in_sam3d:
        print(f"Only in SAM 3D Body: {len(only_in_sam3d)} files")
    if only_in_unidepth:
        print(f"Only in UniDepth: {len(only_in_unidepth)} files")

    if len(common_files) == 0:
        print("Error: No common HDF5 filenames")
        return

    os.makedirs(config.output_root, exist_ok=True)

    total_stats = {
        'processed': 0,
        'skipped': 0,
        'skipped_existing': 0,
        'sam3d_frames': 0,
        'unidepth_frames': 0,
        'matched_frames': 0,
        'existing_frames': 0,
        'valid_rotation_frames': 0,
        'post_convert_valid_frames': 0,
        'final_frames': 0,
        'rotation_errors': {
            'left_wrist_T': 0,
            'right_wrist_T': 0,
            'transformation': 0
        },
        'post_convert_rotation_errors': {
            'left_wrist_T': 0,
            'right_wrist_T': 0,
            'transformation': 0
        }
    }

    for filename in tqdm(common_files, desc="Processing HDF5 files"):
        sam3d_path = os.path.join(config.sam3d_root, filename)
        unidepth_path = os.path.join(config.unidepth_root, filename)
        output_path = os.path.join(config.output_root, filename)

        stats = merge_single_hdf5(sam3d_path, unidepth_path, output_path, config)

        if stats['skipped']:
            total_stats['skipped'] += 1
            if stats.get('skipped_existing', False):
                total_stats['skipped_existing'] += 1
        else:
            total_stats['processed'] += 1
            total_stats['sam3d_frames'] += stats['sam3d_frames']
            total_stats['unidepth_frames'] += stats['unidepth_frames']
            total_stats['matched_frames'] += stats['matched_frames']
            total_stats['existing_frames'] += stats['existing_frames']
            total_stats['valid_rotation_frames'] += stats['valid_rotation_frames']
            total_stats['post_convert_valid_frames'] += stats['post_convert_valid_frames']
            total_stats['final_frames'] += stats['final_frames']
            for key in stats['rotation_errors']:
                total_stats['rotation_errors'][key] += stats['rotation_errors'][key]
            for key in stats['post_convert_rotation_errors']:
                total_stats['post_convert_rotation_errors'][key] += stats['post_convert_rotation_errors'][key]

    print_summary(total_stats, config)


def print_summary(stats: Dict, config: Config) -> None:
    """Print aggregate processing and rotation-validation statistics.

    Args:
        stats (dict): Accumulated frame counts, skips and rotation errors.
        config (Config): Paths, filtering options and rotation tolerances.

    Returns:
        None: Writes the summary to the current output stream.
    """
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Successfully processed: {stats['processed']} files")
    print(f"Skip: {stats['skipped']} files")
    print(f"  Already existing: {stats['skipped_existing']} files")
    print("-" * 60)
    print(f"sam3d total frames: {stats['sam3d_frames']}")
    print(f"unidepth total frames: {stats['unidepth_frames']}")
    print(f"Matched frames: {stats['matched_frames']}")
    print(f"Existing frames: {stats['existing_frames']}")
    print(f"Frames with valid rotations before axis conversion: {stats['valid_rotation_frames']}")
    print(f"Frames with valid rotations after axis conversion: {stats['post_convert_valid_frames']}")
    print(f"Final frames: {stats['final_frames']}")
    print("-" * 60)
    print("Rotation errors before axis conversion:")
    print(f"  left_wrist_T invalid: {stats['rotation_errors']['left_wrist_T']}")
    print(f"  right_wrist_T invalid: {stats['rotation_errors']['right_wrist_T']}")
    print(f"  transformation invalid: {stats['rotation_errors']['transformation']}")
    print("-" * 60)
    print("Rotation errors after axis conversion:")
    print(f"  left_wrist_T invalid: {stats['post_convert_rotation_errors']['left_wrist_T']}")
    print(f"  right_wrist_T invalid: {stats['post_convert_rotation_errors']['right_wrist_T']}")
    print(f"  transformation invalid: {stats['post_convert_rotation_errors']['transformation']}")
    print("-" * 60)
    if stats['matched_frames'] > 0:
        ratio = stats['final_frames'] / stats['matched_frames'] * 100
        print(f"Valid matched-frame fraction: {ratio:.2f}%")
    print("-" * 60)
    print(f"Axis conversion: {'enabled' if config.convert_coordinate_system else 'disabled'}")
    if config.convert_coordinate_system:
        print("  OpenCV (+X right, +Y down, +Z forward) -> Robot (+X forward, +Y left, +Z up)")
    print("-" * 60)
    print("Output format:")
    print("  Position: (x, y, z)")
    print("  Rotation: 6D representation (First two rotation columns in sequence)")
    print("=" * 60)


def main():

    """Run the configured processing stage.

    Returns:
        None: Results are written to the configured output directories.
    """
    config = Config()

    config.sam3d_root = "data/ego4d_fho/v2/hdf5s/sam-3d-body"
    config.unidepth_root = "data/ego4d_fho/v2/hdf5s/unidepth_overlap1"

    config.output_root = "data/ego4d_fho/v2/hdf5s/merged_overlap1"

    config.log_dir = "logs/merge"

    config.check_frame_existence = True
    config.check_rotation_matrix = True
    config.skip_existing = True

    config.convert_coordinate_system = True

    config.ortho_tolerance = 1e-4
    config.det_tolerance = 1e-4

    config.verbose = True

    logger = Logger(config.log_dir)
    sys.stdout = logger

    try:

        print("=" * 60)
        print("HDF5 merge utility (sam-3d-body + unidepth)")
        print("=" * 60)
        print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {logger.log_path}")
        print("-" * 60)
        print(f"sam-3d-body directory: {config.sam3d_root}")
        print(f"unidepth directory: {config.unidepth_root}")
        print(f"Output directory: {config.output_root}")
        print(f"Check frame existence: {config.check_frame_existence}")
        print(f"Check rotation matrices: {config.check_rotation_matrix}")
        print(f"Skip existing outputs: {config.skip_existing}")
        print(f"Axis conversion: {config.convert_coordinate_system}")
        if config.convert_coordinate_system:
            print("  From OpenCV (+X right, +Y down, +Z forward)")
            print("  To Robot (+X forward, +Y left, +Z up)")
        print(f"Orthogonality tolerance: {config.ortho_tolerance}")
        print(f"Determinant tolerance: {config.det_tolerance}")
        print("-" * 60)
        print("Output format:")
        print("  Position: (x, y, z) - 3 dimensions")
        print("  Rotation: 6D representation - First two rotation columns in sequence, 6 values")
        print("=" * 60)

        process_all_hdf5(config)

    finally:

        sys.stdout = logger.terminal
        logger.close()
        print(f"\nLog saved to: {logger.log_path}")

if __name__ == '__main__':
    main()
