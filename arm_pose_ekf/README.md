# arm_pose_ekf

Design scaffold for replacing the current per-frame optimizer plus separate smoothing with one extended Kalman filter.

The ROS package name is lowercase, `arm_pose_ekf`, because ROS package names should be lowercase.

## State

Use joint position and velocity:

```text
x = [q, dq]
q  = [elv_angle, shoulder_elv, shoulder_rot, elbow_flexion,
      forearm_prosup, wrist_x, wrist_z]
dq = joint velocities for the same 7 DOF
```

Constant-velocity prediction:

```text
q_k+1  = q_k + dq_k * dt
dq_k+1 = dq_k
```

The process covariance controls how quickly the filter is allowed to move away from the previous arm state.

## Measurements

The filter can combine several measurements in one update. Each measurement has its own covariance, which replaces the current optimizer weights.

### Robot Hand Pose

If the robot gives the human hand pose in the known shoulder frame:

```text
z_hand = T_shoulder_hand_measured
h_hand(x) = FK_hand(q)
r_hand = log( z_hand^-1 * FK_hand(q) )
```

Use a 6D residual:

```text
r_hand = [position_error, rotation_error]
```

The EKF measurement Jacobian is the hand frame Jacobian from Pinocchio:

```text
H_hand = [J_hand(q), 0]
```

where `J_hand` maps joint velocity to hand spatial velocity.

### Robot Hand Velocity

If the robot also gives hand linear/angular velocity:

```text
z_twist = measured hand twist
h_twist(x) = J_hand(q) * dq
r_twist = z_twist - h_twist(x)
```

The first-order Jacobian can start as:

```text
H_twist = [0, J_hand(q)]
```

For higher accuracy, include the derivative of `J(q) * dq` with respect to `q`, but the simplified version is often a good first implementation.

### Vision Shoulder, Elbow, Wrist Points

If `/joint_positions` provides rough 3D points in camera frame, first transform them into the shoulder frame.

For landmark-style measurements:

```text
z_elbow = measured elbow point
h_elbow(q) = FK_elbow_landmark(q)
r_elbow = z_elbow - h_elbow(q)
H_elbow = [J_elbow_linear(q), 0]
```

The same applies to wrist and shoulder landmarks.

### Link Sphere or Capsule Measurements

The rough 3D points do not need to match an exact URDF frame. They can instead define a soft tube around each link.

For a measured point `p_meas` and modeled link capsule between points `a(q)` and `b(q)`:

```text
c(q) = closest point on segment a(q)-b(q) to p_meas
d(q) = ||p_meas - c(q)|| - capsule_radius
r_capsule = d(q)
```

This is a scalar residual. If `n` is the unit direction from the capsule surface toward the measured point:

```text
H_capsule ~= [ -n^T * J_c(q), 0 ]
```

This behaves like a soft sphere or pipe constraint: the link is encouraged to pass through the measured 3D region without requiring the measurement to land exactly on a named frame.

## EKF Update

Stack all enabled residuals:

```text
r = [r_q, r_hand, r_twist, r_elbow, r_wrist, r_capsule...]
H = stacked measurement Jacobians
R = block diagonal measurement covariance
```

Then:

```text
S = H P H^T + R
K = P H^T S^-1
x = x + K r
P = (I - K H) P (I - K H)^T + K R K^T
```

Because hand pose and capsule measurements are nonlinear, use an iterated EKF update:

```text
for i in 1..N:
  compute FK/Jacobians at current q
  compute residual r
  solve EKF correction
  apply correction to x
```

## Tuning Meaning

Small measurement sigma means "trust this measurement more."

Examples:

```text
hand_pose.sigma_position smaller -> follow robot-measured hand position more
hand_pose.sigma_rotation smaller -> follow robot-measured hand orientation more
keypoint distribution sigma smaller -> keep elbow/wrist near camera-observed points more
process_noise.sigma_acc larger -> allow faster motion and less smoothing
process_noise.sigma_acc smaller -> smoother but more lag
```

This replaces the current optimizer tuning idea:

```text
pos_weight / rot_weight / joint_penalty_weight / velocity_weight / acceleration_weight
```

with covariance terms that have clearer probabilistic meaning.
