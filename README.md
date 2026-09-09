# ActiveMimic Data Pipeline

Data processing code for [ActiveMimic: Egocentric Video Pretraining with Active Perception](https://arxiv.org/abs/2606.06194).

This repository extracts camera and wrist geometry from egocentric video frames and combines the results into HDF5 files. It includes data processing scripts and model integration patches. Policy training, LeRobot conversion, model weights, datasets, and video filtering/segmentation are outside this repository's scope.

## Installation

Use Python 3.10+ and the PyTorch/CUDA environment required by each upstream model:

- [VGGT](https://github.com/facebookresearch/vggt)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)

The `project/` folders contain files to add to the upstream checkouts. For each model:

1. Keep a copy of this repository's `project/<model>/` files.
2. Prepare a complete upstream checkout at `project/<model>`, using the `base_commit` in `upstream_versions.json`.
3. From the release root, apply the matching integration patch:

   ```bash
   git -C project/<model> apply --check ../../patches/<model>-original-runtime.patch
   git -C project/<model> apply ../../patches/<model>-original-runtime.patch
   ```

4. Copy the saved `process_data/` folder into the checkout. For SAM, also copy the saved `notebook/utils.py` over the upstream file. This SAM-specific overlay is installed separately from the runtime patch.
5. Install the model's dependencies and download the required weights according to its upstream instructions.

Apply each patch once to the corresponding upstream revision.

DINOv3 backbone configurations load the official DINOv3 implementation through PyTorch Hub. The first load requires GitHub access or a prepared Torch Hub cache; model weights are loaded from the SAM checkpoint.

## Directory layout

Run all stages from the release root. Configured paths and HDF5 image paths are relative to this directory.

```text
project/{vggt,UniDepth,sam-3d-body}/    # Upstream checkouts with integration files
data/ego4d_fho/v2/segments/           # Input RGB frames
data/ego4d_fho/v2/hdf5s/              # Stage outputs
model/VGGT-1B/
model/unidepth-v2-vitl14/
model/sam-3d-body-vith/               # model.ckpt and assets/mhr_model.pt
model/ViTDet/model_final_f05665.pkl
model/moge-2-vitl-normal/model.pt
logs/merge/
```

Set input and output locations in each launcher. SAM's core checkpoint folder is configured in `project/sam-3d-body/notebook/utils.py`; ViTDet and MoGe locations are configured in the SAM runtime patch. Update these locations when using a different weight directory layout.

## Input

Extract videos into RGB frame folders with temporally sortable filenames:

```text
<INPUT_ROOT>/<video_name>/000000.jpg
<INPUT_ROOT>/<video_name>/000001.jpg
...
```

Use the same image files and relative path base for SAM and VGGT. The SAM launcher uses `frame_interval=1`; prepare frames at the desired sampling rate before running the pipeline. UniDepth reads the frame paths stored by VGGT, and merge matches these paths with SAM's frame paths.

## Run

Use the corresponding model environment for each command, keeping the release root as the working directory:

```bash
bash project/sam-3d-body/process_data/generate_wrist_pose_gpus.sh
bash project/vggt/process_data/generate_cam_pose.sh
bash project/UniDepth/process_data/generate_meter_transformation_log.sh
```

The launchers distribute videos across the available GPUs. The default data flow is:

| Stage | Input | Output directory under `data/ego4d_fho/v2/hdf5s/` |
| --- | --- | --- |
| SAM | RGB frame folders | `sam-3d-body` |
| VGGT | The same RGB frame folders | `vggt_overlap1` |
| UniDepth | VGGT HDF5 files | `unidepth_overlap1` |
| merge | SAM and UniDepth HDF5 files | `merged_overlap1` |

Use a separate output directory for each data-generation run. Configure `REFERENCE_ROOT`, `OUTPUT_ROOT`, `MODEL_PATH`, `SCRIPT_PATH`, and `LOG_DIR` in the launchers as needed.

Set the paths and filtering options in `merge_6D.py:main`, then run:

```bash
python merge_6D.py
```

## Geometry and output

`T_A_B` maps points from frame B to frame A. Model-stage coordinates use OpenCV axes: right, down, forward.

1. **SAM** writes `frame_path`, `left_wrist_T`, and `right_wrist_T` for retained detections. Wrist transforms map wrist coordinates to the current camera frame, with translations in meters.
2. **VGGT** processes overlapping windows `[1..8]`, `[8..15]`, `[15..22]`, and so on. It computes `s = median(previous_shared_depth / current_shared_depth)` over valid pixels, scales the new window's depth and extrinsic translation, and aligns its reference using `E_aligned_k = E_local_k @ inverse(E_local_shared) @ E_stitched_shared`. Each image appears once in the output.
3. VGGT writes `frame_path`, `transformation`, and `depth`. `transformation` contains world-to-camera extrinsics in the first window's predicted reference and normalized scale. Depth is camera-frame depth in that same scale.
4. **UniDepth** estimates metric depth, computes a video-level median scale, and writes `frame_path` and `transformation`, with extrinsic translations in meters.
5. **merge** matches frame paths, inverts camera extrinsics into camera-to-world poses, validates rotations, converts coordinate axes, and writes the following datasets:

   ```text
   frame_path
   lwrist_pos_cami, lwrist_rot6d_cami
   rwrist_pos_cami, rwrist_rot6d_cami
   cam_pos_cam1, cam_rot6d_cam1
   ```

Merged coordinates use robot axes: forward, left, up. Positions are in meters. Camera poses use the first VGGT window's predicted reference; wrist poses use their corresponding current-camera frames.

Each rotation is encoded as the first two matrix columns in sequence: `[r11,r21,r31,r12,r22,r32]`.
