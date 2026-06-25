# CLAUDE.md — arm_pose_pf

Guidance for Claude Code working inside this package. See the workspace-level
`../CLAUDE.md` for the big picture and the shared 7-DOF arm state.

## What this package is

A single ROS 2 node (`arm_pose_pf_node`) that estimates the 7-DOF human arm state
`x = [q, dq]` (14-D) with a **bootstrap particle filter**. It is the PF counterpart
to `arm_pose_ekf` and is meant to stay measurement/frame-compatible with it.

Everything lives in three files:
- `src/arm_pose_pf_node.cpp` (~1.5k lines) — all logic
- `include/arm_pose_pf/arm_pose_pf_node.hpp` — the `ArmPosePfNode` class + nested `Pf` struct (particles, weights, resample)
- `config/arm_pose_pf.yaml` — every tunable; nothing is hard-coded

## Build / run

```bash
cd ~/pose_ws
colcon build --packages-select arm_pose_pf
source install/setup.bash
ros2 run arm_pose_pf arm_pose_pf_node --ros-args \
  --params-file install/arm_pose_pf/share/arm_pose_pf/config/arm_pose_pf.yaml
```

Built with C++17 and **OpenMP** (per-particle FK runs in parallel). Depends on
Pinocchio, tf2, yaml-cpp.

## Actual I/O (the README is partly aspirational — trust this)

Subscribes:
- `topics.robot_joint_states` (`/lbr/joint_states`, `sensor_msgs/JointState`) — **the trigger**; `robotJointCallback` is the only thing that advances predict + correct.
- `topics.forward_solver_joint_states` (`/arm/joint_states`, `std_msgs/Float32MultiArray`) — the vision input, a **q vector from `image_pose_tracking`'s forward solver**, cached by `forwardSolverJointCallback`.

Publishes: `topics.output_joint_states` (`/arm_pose_pf/joint_states`, `sensor_msgs/JointState`).

**Important:** despite what `README.md` says, this node does **not** subscribe to the
`/keypoint_distributions/*` Gaussian topics. Vision enters only as the forward-solver
q vector. Treat the README's "Shared Input" section as the design intent, not the code.

## How a cycle actually works

1. **Predict** (`predictTo`) — constant-velocity with velocity decay and sampled
   acceleration noise (`process.acceleration_variance`, `velocity_decay`).
2. **Correct from the robot** (`correctHandPoseAndTwist`) — the robot EE pose/twist,
   mapped through the `ee_to_hand` hand-eye transform into the shoulder frame, is the
   **primary** correction. Each particle is scored by `handMeasurementLikelihood`
   (hand position / rotation / twist, toggled by `use_hand_pose_*` / `use_hand_twist_`).
   This runs concurrently across particles using one Pinocchio `Data` per OpenMP thread
   (`arm_data_pool_`) — `arm_model_` is read-only and shared; never write it from a worker.
3. **Vision** enters three different ways (do not conflate them):
   - **Initialization** (`initializeParticlesFromForwardSolver`) — seed particles around the first forward-solver q.
   - **Proposal injection** (`injectForwardSolverProposalParticles`) — each resample, a fraction of particles is redrawn near the latest forward-solver q. *This*, not a likelihood, is what lets redundant / null-space joints track and denoise over time.
   - **Optional robust likelihood** (`correctVisionQ`) — a Student-t likelihood on q with NIS gating, **off by default** (`use_vision_measurement: false`).
4. **Resample** when effective sample size drops below
   `resample_effective_ratio * particle_count`, then add `roughening_stddev` jitter.
5. **Constraints** (`applyStateConstraints`) — URDF position/velocity box limits, read
   from the rendered arm xacro via Pinocchio, projected onto every particle each step.

## Things that are easy to get wrong

- **Two models, two URDFs.** `robot_model_` is the manipulator (e.g. `urdf/iiwa7/iiwa7.urdf`)
  used for robot EE FK; `arm_model_` is the rendered human-arm xacro (`arm_xacro_file`,
  shared from `image_pose_tracking/config`) used for the 7-DOF state. Joint limits and the
  mimic-joint multiplier come from the arm model.
- **The robot pose/twist is the only correction by default**, not vision. If proximal /
  axial joints look untracked, the lever is `forward_solver_proposal`, not a sigma.
- **OpenMP safety:** anything touched inside the per-particle loop must use the
  thread-local `Data` from `arm_data_pool_`; sharing `arm_data_` across threads is a bug.
- **`max_update_rate_hz` / queue depths** are deliberately small to drop stale robot
  callbacks rather than accumulate lag — keep them below the robot joint-state rate.
- Keep tunables in `config/arm_pose_pf.yaml` and aligned with `arm_pose_ekf` where the two
  share meaning, so the backends stay swappable in experiments.
