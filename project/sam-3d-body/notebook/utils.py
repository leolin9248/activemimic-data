"""
SAM 3D Body notebook overlay for the ActiveMimic data pipeline.

Derived from facebookresearch/sam-3d-body.
Upstream source and license: https://github.com/facebookresearch/sam-3d-body
Copy this file over the upstream notebook/utils.py after applying the SAM runtime
patch. Model paths are relative to the ActiveMimic release root.
"""

import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def setup_sam_3d_body_local(
    ckpt_folder: str = "model/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """Load the local SAM 3D Body model and optional estimators.

    Args:
        ckpt_folder (str): Local folder containing model.ckpt and assets/mhr_model.pt.
        detector_name (str): Detector backend name; an empty value disables detection.
        segmentor_name (str): Segmentor backend selected when segmentor_path is nonempty.
        fov_name (str): FOV backend name; an empty value selects the estimator default.
        detector_path (str): Reserved interface parameter; configure weights in the corresponding backend loader.
        segmentor_path (str): Optional segmentor path; an empty value disables segmentation.
        fov_path (str): Reserved interface parameter; configure weights in the corresponding backend loader.
        device (torch.device or str): Device used for inference.

    Returns:
        SAM3DBodyEstimator: Estimator configured for inference.
    """
    print(f"Loading SAM 3D Body model from {ckpt_folder}...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_cfg = load_sam_3d_body(checkpoint_path=f"{ckpt_folder}/model.ckpt",
                                        mhr_path=f"{ckpt_folder}/assets/mhr_model.pt",
                                        device=device)

    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device)

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator


def setup_sam_3d_body(
    hf_repo_id: str = "facebook/sam-3d-body-vith",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    device: str = "cuda",
):
    """Load SAM 3D Body from the configured local checkpoint folder.

    Args:
        hf_repo_id (str): Model label displayed during loading. The checkpoint folder is configured below.
        detector_name (str): Detector backend name; an empty value disables detection.
        segmentor_name (str): Segmentor backend selected when segmentor_path is nonempty.
        fov_name (str): FOV backend name; an empty value selects the estimator default.
        detector_path (str): Reserved interface parameter; configure weights in the corresponding backend loader.
        segmentor_path (str): Optional segmentor path; an empty value disables segmentation.
        fov_path (str): Reserved interface parameter; configure weights in the corresponding backend loader.
        device (torch.device or str): Device used for inference.

    Returns:
        SAM3DBodyEstimator: Estimator configured for inference.
    """
    print(f"Loading SAM 3D Body model from {hf_repo_id}...")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_folder = "model/sam-3d-body-vith"
    ckpt_path = f"{ckpt_folder}/model.ckpt"
    model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_path,
                                        mhr_path=f"{ckpt_folder}/assets/mhr_model.pt")

    human_detector, human_segmentor, fov_estimator = None, None, None

    if detector_name:
        print(f"Loading human detector from {detector_name}...")
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(name=detector_name, device=device)

    if segmentor_path:
        print(f"Loading human segmentor from {segmentor_path}...")
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name, device=device, path=segmentor_path
        )

    if fov_name:
        print(f"Loading FOV estimator from {fov_name}...")
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=fov_name, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    print(f"Setup complete!")
    print(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    print(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    print(f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}")

    return estimator


def setup_visualizer():
    """Configure a skeleton visualizer using MHR70 metadata.

    Returns:
        SkeletonVisualizer: Visualizer with the selected pose metadata.
    """
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Render keypoints and bounding boxes for each person.

    Args:
        img_cv2 (np.ndarray): BGR image with shape (H, W, 3).
        outputs (list[dict]): Per-person SAM predictions for the input image.
        visualizer (SkeletonVisualizer): Visualizer configured with MHR70 keypoint metadata.

    Returns:
        list[np.ndarray]: One BGR visualization of shape (H, W, 3) per person.
    """
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Render image, overlay, front and side mesh views for each person.

    Args:
        img_cv2 (np.ndarray): BGR image with shape (H, W, 3).
        outputs (list[dict]): Per-person SAM predictions for the input image.
        faces (np.ndarray): Mesh triangle vertex indices with shape (F, 3).

    Returns:
        list[np.ndarray]: Four-view images with shape (H, 4 * W, 3).
    """
    results = []

    for pid, person_output in enumerate(outputs):

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        img_orig = img_cv2.copy()

        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def save_mesh_results(
    img_cv2: np.ndarray,
    outputs: List[Dict[str, Any]],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> List[str]:
    """Export meshes, camera focal length and preview images.

    Args:
        img_cv2 (np.ndarray): BGR image with shape (H, W, 3).
        outputs (list[dict]): Per-person SAM predictions for the input image.
        faces (np.ndarray): Mesh triangle vertex indices with shape (F, 3).
        save_dir (str): Directory for exported meshes and preview images.
        image_name (str): Filename stem used for exported artifacts.

    Returns:
        list[str]: Paths of the exported PLY meshes.
    """
    import json

    os.makedirs(save_dir, exist_ok=True)
    ply_files = []

    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(save_dir, f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        print(f"Saved focal length: {focal_length_path}")

    for pid, person_output in enumerate(outputs):

        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(save_dir, mesh_filename)
        tmesh.export(mesh_path)
        ply_files.append(mesh_path)

        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, overlay_filename), img_mesh_overlay)

        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        cv2.imwrite(os.path.join(save_dir, bbox_filename), img_bbox)

        print(f"Saved mesh: {mesh_path}")
        print(f"Saved overlay: {os.path.join(save_dir, overlay_filename)}")
        print(f"Saved bbox: {os.path.join(save_dir, bbox_filename)}")

    return ply_files


def display_results_grid(
    images: List[np.ndarray], titles: List[str], figsize_per_image: tuple = (6, 6)
):
    """Display result images with corresponding titles.

    Args:
        images (list[np.ndarray]): Images displayed in the result grid.
        titles (list[str]): Titles corresponding to the displayed images.
        figsize_per_image (tuple[float, float]): Width and height allocated per image, in inches.

    Returns:
        None: Displays a Matplotlib figure when images are available.
    """
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return

    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )

    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:

            if img.dtype == np.uint8 and np.mean(img) > 1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            img_rgb = img

        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def process_image_with_mask(estimator, image_path: str, mask_path: str):
    """Infer body predictions using a mask-derived bounding box.

    Args:
        estimator (SAM3DBodyEstimator): Configured body estimator.
        image_path (str): Input image path relative to the release root.
        mask_path (str): Grayscale mask path; pixels above 127 are foreground.

    Returns:
        list[dict]: Per-person predictions, or an empty list for an empty mask.
    """

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    mask_binary = (mask > 127).astype(np.uint8) * 255

    print(f"Processing image with external mask: {mask_path}")
    print(f"Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")

    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        print("Warning: Mask is empty, no objects detected")
        return []

    x, y, w, h = cv2.boundingRect(coords)
    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    print(f"Computed bbox from mask: {bbox[0]}")

    outputs = estimator.process_one_image(image_path, bboxes=bbox, masks=mask_binary)

    return outputs
