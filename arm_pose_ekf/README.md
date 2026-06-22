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

## As Implemented (read this before tuning the math)

The sections above are the original design scaffold. The actual node
(`src/arm_pose_ekf_node.cpp`) implements a focused subset, applied as
**sequential** corrections (not one stacked update, and not iterated). The arm
kinematics are re-linearized between every step via `updateArmKinematics`.

**Trigger:** `robotJointCallback` is the only thing that advances the filter.
Keypoint callbacks merely cache the latest shoulder-frame Gaussian.

**Prediction** — pure constant velocity with white-noise acceleration:

```text
q  <- q + dt * dq
dq <- dq
P  <- F P F^T + Q     (Q = white-noise-acceleration covariance)
```

**Corrections actually run, in order:**

1. Hand position — `r = p_meas - p_model(q)`, `H = [J_v(q), 0]` (3D).
2. Hand rotation — `r = log3(R_meas * R_model^T)`, `H = [J_w(q), 0]` (3D).
   Left/world-aligned error to match the LOCAL_WORLD_ALIGNED Jacobian.
3. Hand twist — `r = v_meas - J_hand(q) * dq`, `H = [0, J_hand(q)]` (6D).
4. Bone-direction keypoints — on the **unit** bone vector
   `u = (p_end - p_start)/||p_end - p_start||`:

   ```text
   r = u_meas - u_model(q)
   H = [ (I - u u^T)/||s|| * (J_end - J_start) , 0 ]
   ```

   Two segments, each masked to only the DOFs that place that bone:
   - upper_arm (shoulder->elbow): DOFs {0,1} = elv_angle, shoulder_elv
   - forearm (elbow->wrist): DOFs {2,3} = shoulder_rot, elbow_flexion

   Note: shoulder axial rotation (DOF 2) gets keypoint information **only**
   from the forearm direction, and the wrist DOFs {5,6} and forearm
   pronation (DOF 4) get **no** keypoint correction at all — they are driven
   solely by the hand pose/twist. This is the most likely source of the
   steady-state offsets seen in `shoulder_rot`, `elbow_roty`, and `wrist_rotz`.

The hand rotation correction supports an **anisotropic** measurement covariance
(`rotation_axis_ratio`, in hand-frame axes) to model grip compliance: rotation
about a grip-compliant axis can be down-weighted so it barely corrects the
wrist/forearm. `[1,1,1]` is plain isotropic.

**Occlusion-gated proximal anchor.** When no keypoint update runs for a cycle
(elbow occluded -> `keypointSegmentsTooClose`, or vision stale/missing), the
filter would otherwise let the 6-DOF hand pose drift the proximal null space —
shoulder axial rotation especially, which the hand pose cannot uniquely resolve.
Instead, `applyProximalAnchor` holds the four vision-observed proximal joints
`q0-q3` near their last vision-confident value via a soft equality
pseudo-measurement (`r = q_anchor - q`, `H = [I4, 0]`, `R = sigma^2 I`). The
anchor value is cached on every cycle where a real keypoint correction ran. This
is information, not a hand-set covariance: it collapses `P` on `q0-q3` the same
way vision did, so the hand pose can no longer move them, and it pulls back drift
that already started.

The anchor target **coasts** rather than freezing: each occluded cycle it is
advanced by `dt * dq` using the hand-twist-corrected velocity (the twist update
is not vision-gated, so `dq` stays honest through the gap). This makes it a
constant-velocity hold consistent with the predict step — it cancels only the
hand-pose null-space leak while letting twist + dynamics carry the proximal
joints through a 1-2 s occlusion, instead of pinning them at a stale snapshot.
Tuned via `proximal_anchor_sigma` (radians; smaller = stiffer hold) and toggled
by `proximal_anchor_enabled`. This is a pseudo-measurement (cf. the zero-velocity
update / ZUPT in inertial navigation).

**Not implemented** (mentioned in the scaffold above but absent in code): a
joint prior residual `r_q`, capsule/sphere link measurements, and the iterated
EKF loop. The keypoint update uses bone *directions* rather than landmark
positions, so it constrains arm orientation but not absolute segment length.

**State constraints:** after every predict/correct, `applyStateConstraints`
enforces the URDF position/velocity box limits via **Gaussian PDF truncation**
(Simon & Simon 2010). For each box-constrained component it applies the
closed-form rank-1 update `x += (m/s)·v`, `P += ((w-1)/s²)·v·vᵀ`, where
`v = P.col(i)`, `s² = P(i,i)`, and `(m, w)` are the mean/variance of a standard
normal truncated to the standardized bounds. This updates the covariance and
propagates to correlated states (incl. q↔dq), and is a no-op when the marginal
already sits several sigma inside the bounds. `dq` is truncated to its velocity
limits, and the "no outward velocity at an active position bound" rule is
applied as a one-sided velocity truncation.
