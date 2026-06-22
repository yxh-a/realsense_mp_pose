# arm_pose_pf

`arm_pose_pf` estimates the same 7-DOF human arm state as `arm_pose_ekf`, but uses a bootstrap particle filter instead of an extended Kalman filter.

```text
x = [q, dq]
q  = [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion,
      forearm_prosup, wrist_x, wrist_z]
dq = joint velocities for the same 7 DOF
```

## Shared Input And Frame Math

The node intentionally mirrors `arm_pose_ekf` for the measurement inputs and frame transformations:

- subscribes to the same robot joint-state topic
- subscribes to the same `/keypoint_distributions/{right_wrist,right_elbow,right_shoulder,left_shoulder}` topics
- renders the same configurable arm xacro and loads it through Pinocchio
- uses the same `ee_to_hand` calibration transform
- looks up the same shoulder, world, and robot end-effector TF frames
- transforms keypoint Gaussian means and covariances into the shoulder frame with the same rotation/covariance math
- converts robot end-effector twist into measured hand twist in the shoulder frame with the same cross-product offset
- publishes the same seven arm joint names as a `sensor_msgs/JointState`

The default output topic is `/arm_pose_pf/joint_states`.

## Particle Filter

Each particle carries one complete `[q,dq]` hypothesis. Prediction uses the same constant-velocity model and velocity decay idea as `arm_pose_ekf`, with sampled acceleration noise:

```text
dq_k+1 = exp(-decay * dt) * dq_k + dt * a
q_k+1  = q_k + dt * exp(-decay * dt) * dq_k + 0.5 * dt^2 * a
a      ~ N(0, acceleration_variance)
```

Measurement updates do not linearize with Jacobians. Instead, each particle is scored by the same residual definitions used by the EKF:

- hand position residual: measured hand position minus Pinocchio FK hand position
- hand rotation residual: left/world-aligned SO(3) error `log3(R_measured * R_model^T)`
- hand twist residual: measured hand twist minus `J_hand(q) * dq`
- keypoint residuals: measured upper-arm and forearm unit bone directions minus model unit bone directions

Weights are normalized after each enabled measurement group. The filter resamples when effective particle count falls below `particle_filter.resample_effective_ratio * particle_count`, then adds a small `particle_filter.roughening_stddev` jitter to preserve diversity.

## Configuration

The main config file is:

```text
config/arm_pose_pf.yaml
```

Particle-specific parameters:

```yaml
particle_filter:
  particle_count: 500
  seed: 1
  resample_effective_ratio: 0.5
  roughening_stddev: 1.0e-4
```

The measurement, TF, xacro, joint-limit, and topic parameters are kept aligned with `arm_pose_ekf` so the two packages can be swapped during experiments.

## Build

```bash
colcon build --packages-select arm_pose_pf
```

Run the node with your usual ROS 2 launch setup or directly:

```bash
ros2 run arm_pose_pf arm_pose_pf_node --ros-args --params-file install/arm_pose_pf/share/arm_pose_pf/config/arm_pose_pf.yaml
```
