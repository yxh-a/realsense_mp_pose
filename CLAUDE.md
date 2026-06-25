# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A ROS 2 workspace (`pose_ws/src`) for **estimating a 7-DOF human arm pose** by fusing
camera-based keypoint detection with robot-measured hand pose. The robot (a KUKA LBR,
frame prefix `lbr_`) holds the human's hand; vision provides rough 3D keypoints; the
estimators reconcile both against a kinematic model of the human arm.

The 7 DOF (an OpenSim shoulder model, rendered to URDF and loaded via **Pinocchio**) are:
```
q = [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion, forearm_prosup, wrist_x, wrist_z]
```

There are **two interchangeable estimation backends** over the same state and the same
measurement/frame math, kept deliberately aligned so they can be swapped in experiments:
- `arm_pose_ekf` — sequential extended Kalman filter (the maintained primary backend; see its README)
- `arm_pose_pf` — bootstrap particle filter

## Do not touch (imported / vendored)

These are imported third-party packages — **do not modify them**:
- `apriltag_ros`, `apriltag_msgs` — AprilTag detection
- `easy_handeye2` — hand-eye calibration
- `arm_moveit_config` — generated MoveIt config
- `realsense-ros` — a git submodule (see `.gitmodules`)

Edit only the custom packages: `image_pose_tracking`, `arm_pose_ekf`, `arm_pose_pf`,
`ik_solver_moveit`.

## Build, run, test

This directory is `src`; **colcon runs from the workspace root one level up** (`pose_ws`).

```bash
cd ~/pose_ws
colcon build --packages-select arm_pose_ekf      # build one package
colcon build                                     # build everything
source install/setup.bash                        # required before ros2 run
```

Run an estimator with its params file (note the `install/.../config` path, not `src`):
```bash
ros2 run arm_pose_ekf arm_pose_ekf_node --ros-args \
  --params-file install/arm_pose_ekf/share/arm_pose_ekf/config/arm_pose_ekf.yaml
ros2 run arm_pose_pf  arm_pose_pf_node  --ros-args \
  --params-file install/arm_pose_pf/share/arm_pose_pf/config/arm_pose_pf.yaml
```

Python nodes (`image_pose_tracking`, an `ament_python` package) expose console scripts —
e.g. `ros2 run image_pose_tracking keypoint_distribution_extraction` (see `setup.py`
`entry_points` for the full list).

Tests / linting: `colcon test --packages-select <pkg>` then `colcon test-result --verbose`.
Python packages lint via `ament_flake8` / `ament_pep257` (configured in `package.xml`
`test_depend`); C++ lint via `ament_lint_auto`.

## Pipeline / data flow

```
camera RGB ─► mediapipe_kp ─► /keypoints_2d (Int16MultiArray), /pose_annotated
                                    │
depth pointcloud ───────────────────┤
(/camera/camera/depth/color/points) │
                                    ▼
        keypoint_distribution_extraction
   ─► /keypoint_distributions/{right_wrist,right_elbow,right_shoulder,left_shoulder}
      (PoseWithCovarianceStamped — a 3D Gaussian per keypoint in camera frame)
                                    │
   robot /lbr/joint_states + TF + ee_to_hand calibration
                                    ▼
        estimator backend ─► <backend>/joint_states (sensor_msgs/JointState, 7 DOF)
            arm_pose_ekf ─► /arm_pose_ekf/joint_states
            arm_pose_pf  ─► /arm_pose_pf/joint_states
```

Each backend transforms the keypoint Gaussians (mean **and** covariance) into the
shoulder frame, converts the robot end-effector pose/twist into a measured hand
pose/twist via the `ee_to_hand` hand-eye transform, and corrects the modeled arm.

## Things that are easy to get wrong

- **The estimators do not consume keypoint *positions* as landmarks.** They use the
  *unit bone directions* (shoulder→elbow, elbow→wrist), so vision constrains arm
  orientation but not absolute segment length. Segment lengths come from
  `subject_arm_lengths.yaml` keyed by `subject_id` (with fallback lengths in the YAML).
- **Frames are configured, not hard-coded.** `world_frame`, `shoulder_frame`, `hand_frame`,
  `robot_ee_frame`, and the `robot_prefix` live in the params YAML; keep the EKF and PF
  configs aligned when changing them.
- **The arm model is a rendered xacro.** `arm_xacro_file` (e.g.
  `right_arm_osim_shoulder_mesh.urdf.xacro`, shared from `image_pose_tracking/config`) is
  what Pinocchio loads; joint position/velocity limits are read from it, not duplicated in code.
- **`arm_pose_ekf` as-implemented differs from its design scaffold.** Read the
  "As Implemented" section of `arm_pose_ekf/README.md` before touching the filter math
  (sequential not stacked corrections, occlusion-gated proximal anchor, Gaussian-truncation
  state constraints, which DOFs each measurement actually moves).
