"""
Physics Simulation for Betaflight SITL

This module implements the drone physics simulation that interfaces with
Betaflight SITL. Unlike the main drone example, this simulation does NOT
include any control systems - all control comes from Betaflight.

The simulation:
1. Receives motor commands from Betaflight (normalized 0-1)
2. Converts to thrust forces and torques
3. Integrates rigid body dynamics
4. Generates sensor outputs (IMU, position, velocity)

Physics Model:
- 6-DOF rigid body with thrust from 4 motors
- Quadratic drag in linear and angular motion
- Ground collision (simple constraint)
- Motor dynamics (first-order response)
"""

import typing as ty
from dataclasses import dataclass, field

import elodin as el
import jax
import jax.numpy as jnp

from config import DroneConfig


# --- Component Type Definitions ---

# Motor commands from Betaflight (normalized 0-1)
# Marked as external_control so Betaflight can write to it via Elodin-DB
#
# Motor order after SITL's Gazebo remapping (see sitl.c pwmCompleteMotorUpdate):
#   Index 0: Front Right (FR) - CCW  (originally BF Motor 1)
#   Index 1: Back Left (BL) - CCW    (originally BF Motor 2)
#   Index 2: Front Left (FL) - CW    (originally BF Motor 3)
#   Index 3: Back Right (BR) - CW    (originally BF Motor 0)
#
# See config.py for motor positions and spin directions.
MotorCommand = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_command",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={
            "element_names": "FR,BL,FL,BR",  # After SITL Gazebo remapping
            "priority": 100,
            "external_control": "true",  # Allows external writes from Betaflight bridge
        },
    ),
]

# Current motor thrust state (for dynamics)
# Same motor order as MotorCommand: FR(0), BL(1), FL(2), BR(3)
MotorThrust = ty.Annotated[
    jax.Array,
    el.Component(
        "motor_thrust",
        el.ComponentType(el.PrimitiveType.F64, (4,)),
        metadata={"element_names": "FR,BL,FL,BR", "priority": 99},
    ),
]

# Body frame thrust force (for visualization)
BodyThrust = ty.Annotated[
    el.SpatialForce,
    el.Component(
        "body_thrust",
        metadata={"priority": 98, "element_names": "τx,τy,τz,fx,fy,fz"},
    ),
]

# Drag force (for visualization)
BodyDrag = ty.Annotated[
    jax.Array,
    el.Component(
        "body_drag",
        el.ComponentType(el.PrimitiveType.F64, (3,)),
        metadata={"element_names": "fx,fy,fz"},
    ),
]

# Simulation time component
SimTime = ty.Annotated[
    jax.Array,
    el.Component(
        "sim_time",
        el.ComponentType(el.PrimitiveType.F64, (1,)),
        metadata={"priority": 200},
    ),
]

# Wake turbulence noise state (Ornstein-Uhlenbeck process, 6 DOF: fx,fy,fz,tx,ty,tz)
WakeNoise = ty.Annotated[
    jax.Array,
    el.Component(
        "wake_noise",
        el.ComponentType(el.PrimitiveType.F64, (6,)),
        metadata={"element_names": "fx,fy,fz,tx,ty,tz"},
    ),
]


@dataclass
class Drone(el.Archetype):
    """
    Drone archetype with physics state components.

    This archetype is spawned for each simulated drone entity.
    """

    motor_command: MotorCommand = field(default_factory=lambda: jnp.zeros(4))
    motor_thrust: MotorThrust = field(default_factory=lambda: jnp.zeros(4))
    body_thrust: BodyThrust = field(default_factory=lambda: el.SpatialForce())
    body_drag: BodyDrag = field(default_factory=lambda: jnp.zeros(3))
    sim_time: SimTime = field(default_factory=lambda: jnp.zeros(1))
    wake_noise: WakeNoise = field(default_factory=lambda: jnp.zeros(6))


# --- Physics Systems ---


def create_motor_dynamics(config: DroneConfig):
    """
    Create motor dynamics system.

    Motors have first-order response dynamics:
        thrust' = (commanded - thrust) / time_constant
    """
    dt = config.sim_time_step
    tau = config.motor_time_constant
    max_thrust = config.motor_max_thrust
    alpha = dt / (dt + tau)  # First-order filter coefficient

    @el.map
    def motor_dynamics(cmd: MotorCommand, thrust: MotorThrust) -> MotorThrust:
        """Update motor thrust based on commanded values."""
        # Clamp commands to valid range
        cmd_clamped = jnp.clip(cmd, 0.0, 1.0)

        # Convert normalized command to thrust target
        target_thrust = cmd_clamped * max_thrust

        # First-order low-pass filter for motor response
        new_thrust = thrust + alpha * (target_thrust - thrust)

        return new_thrust

    return motor_dynamics


def create_rotor_aero_system(config: DroneConfig):
    """
    Modify per-rotor thrust for aerodynamic effects:
    1. IGE (In-Ground-Effect): thrust efficiency gain near ground (Cheeseman-Bennett)
    2. VRS (Vortex Ring State): thrust loss when descending into own wake

    Runs after motor_dynamics, before body_thrust computation.
    Per-rotor: each motor gets its own AGL for asymmetric IGE on tilted drone.
    """
    motor_positions = jnp.array(config.motor_positions)
    ground_level = config.ground_level
    rotor_radius = config.rotor_radius
    ige_max_gain = config.ige_max_gain
    propwash_loss_max = config.propwash_loss_max
    propwash_peak_ratio = config.propwash_peak_ratio
    propwash_width = config.propwash_width
    air_density = config.air_density
    disk_area = jnp.pi * rotor_radius ** 2  # single rotor disk area
    hover_thrust_per_motor = config.mass * config.gravity / 4.0  # ~1.96N

    # IGE gate configuration (profile-driven, baked in at construction time)
    thrust_gate_params = config.ige_thrust_gate
    has_thrust_gate = thrust_gate_params is not None
    thrust_gate_low, thrust_gate_high = thrust_gate_params if has_thrust_gate else (0.0, 1.0)
    agl_gate_low, agl_gate_high = config.ige_agl_gate

    @el.map
    def rotor_aero(
        thrust: MotorThrust,
        pos: el.WorldPos,
        vel: el.WorldVel,
    ) -> MotorThrust:
        """Apply IGE gain and VRS loss to per-rotor thrust."""
        quat = pos.angular()

        # --- Per-rotor IGE thrust gain ---
        # Transform each motor position to world frame, compute per-rotor AGL.
        # Gates are profile-driven (config.physics_profile):
        #   baseline:  thrust gate 0.65→1.0, AGL gate 0.03→0.08m — CI-stable.
        #   strict:    thrust gate 0.3→0.6,  AGL gate 0.01→0.03m — long-term regression pressure.
        #   realistic: no thrust gate (always 1.0), AGL gate 0.01→0.03m — matches real hardware physics.
        def ige_per_motor(i, modified_thrust):
            pt_body = motor_positions[i]
            pt_world = pos.linear() + quat @ pt_body
            agl = jnp.maximum(pt_world[2] - ground_level, 0.01)  # clamp to avoid div-by-zero

            # Cheeseman-Bennett heuristic: base gain = k * (R/(4*agl))^2
            ratio = rotor_radius / (4.0 * agl)
            base_gain = jnp.clip(ige_max_gain * ratio * ratio, 0.0, ige_max_gain)

            # Thrust gate (Python-time bool — JAX traces only the active branch)
            if has_thrust_gate:
                thrust_ratio = modified_thrust[i] / jnp.maximum(hover_thrust_per_motor, 1e-6)
                t_gate = jnp.clip(
                    (thrust_ratio - thrust_gate_low) / (thrust_gate_high - thrust_gate_low),
                    0.0,
                    1.0,
                )
                thrust_gate = t_gate * t_gate * (3.0 - 2.0 * t_gate)  # smoothstep
            else:
                thrust_gate = 1.0

            # AGL gate: protects contact zone, always present
            a_gate = jnp.clip((agl - agl_gate_low) / (agl_gate_high - agl_gate_low), 0.0, 1.0)
            agl_gate = a_gate * a_gate * (3.0 - 2.0 * a_gate)  # smoothstep

            ige_factor = 1.0 + base_gain * thrust_gate * agl_gate

            return modified_thrust.at[i].set(modified_thrust[i] * ige_factor)

        new_thrust = jax.lax.fori_loop(0, 4, ige_per_motor, thrust)

        # --- VRS thrust loss on descent ---
        # Compute induced velocity from BASE (pre-loss) thrust
        total_base_thrust = jnp.sum(thrust)  # use original, not IGE-modified
        v_induced = jnp.sqrt(jnp.maximum(total_base_thrust, 0.01) / (2.0 * air_density * 4.0 * disk_area))

        # Descent speed along rotor disk normal (body Z), not world Z
        quat_inv = quat.inverse()
        vel_body = quat_inv @ vel.linear()
        descent_body = jnp.maximum(-vel_body[2], 0.0)  # positive when descending in body frame

        # Translation in rotor-disk plane (body XY) for VRS suppression
        # VRS is strongest at low translation; fades with horizontal movement in disk plane
        v_disk_plane = jnp.sqrt(vel_body[0] ** 2 + vel_body[1] ** 2)
        vrs_suppression = 1.0 - jnp.clip(v_disk_plane / jnp.maximum(v_induced, 0.1), 0.0, 1.5) / 1.5

        # Bell-shaped loss: peak at propwash_peak_ratio, Gaussian profile
        descent_ratio = descent_body / jnp.maximum(v_induced, 0.1)
        loss = propwash_loss_max * jnp.exp(
            -((descent_ratio - propwash_peak_ratio) / propwash_width) ** 2
        )
        loss = loss * vrs_suppression

        # Apply uniform loss to all rotors
        new_thrust = new_thrust * (1.0 - loss)

        return new_thrust

    return rotor_aero


def create_body_thrust_system(config: DroneConfig):
    """
    Create system to compute body-frame thrust and torques.

    Each motor produces:
    - Thrust force in body Z direction
    - Torque from thrust offset (roll/pitch)
    - Reaction torque from spin (yaw)
    """
    motor_positions = jnp.array(config.motor_positions)
    thrust_directions = jnp.array(config.motor_thrust_directions)
    spin_directions = jnp.array(config.motor_spin_directions)
    torque_coeff = config.motor_torque_coeff

    # Compute torque arms (cross product of position and thrust direction)
    torque_arms = jnp.cross(motor_positions, thrust_directions)

    @el.map
    def compute_body_thrust(thrust: MotorThrust) -> BodyThrust:
        """Compute total body-frame force and torque from motors."""
        # Linear force: sum of all motor thrusts in their directions
        total_force = jnp.sum(thrust[:, None] * thrust_directions, axis=0)

        # Torque from differential thrust (roll/pitch)
        diff_torque = jnp.sum(thrust[:, None] * torque_arms, axis=0)

        # Yaw torque from motor spin (reaction torque)
        yaw_torque = jnp.sum(thrust * spin_directions) * torque_coeff

        # Combine torques
        total_torque = diff_torque + jnp.array([0.0, 0.0, yaw_torque])

        return el.SpatialForce(linear=total_force, torque=total_torque)

    return compute_body_thrust


def create_drag_system(config: DroneConfig):
    """
    Create aerodynamic drag system.

    Drag is modeled as quadratic:
        F_drag = -0.5 * rho * Cd * A * |v| * v

    Simplified to linear coefficient times v * |v|
    """
    linear_drag = jnp.array(config.linear_drag)

    @el.map
    def compute_drag(vel: el.WorldVel) -> BodyDrag:
        """Compute drag force from velocity."""
        v = vel.linear()
        v_mag = jnp.linalg.norm(v)

        # Quadratic drag: F = -k * |v| * v
        drag_force = -linear_drag * v_mag * v

        return drag_force

    return compute_drag


def create_apply_forces_system(config: DroneConfig):
    """
    Create system to apply all forces to the body.

    Combines:
    - Motor thrust (body frame, rotated to world)
    - Drag (world frame)
    - Gravity (world frame)
    - Multi-point ground contact (spring-damper at 4 landing points)
    - Wake turbulence (OU correlated noise, scales with ground proximity + descent rate)
    """
    gravity_vec = jnp.array([0.0, 0.0, -config.gravity])
    angular_drag = jnp.array(config.angular_drag)
    ge_height = config.ground_effect_height
    ge_force_std = config.ground_effect_force_std
    ge_torque_std = config.ground_effect_torque_std
    pw_descent_force_std = config.propwash_descent_force_std
    pw_descent_torque_std = config.propwash_descent_torque_std
    ground_level = config.ground_level
    air_density = config.air_density
    rotor_radius = config.rotor_radius
    disk_area = jnp.pi * rotor_radius ** 2

    # OU noise parameters
    dt = config.sim_time_step
    tau = config.wake_noise_tau
    ou_decay = jnp.exp(-dt / tau)
    ou_diffusion = jnp.sqrt(1.0 - jnp.exp(-2.0 * dt / tau))

    # Multi-point ground contact: 4 points at arm undersides
    contact_pts_body = jnp.array(config.contact_points)  # (4, 3) in body FLU
    contact_k = config.contact_stiffness
    contact_c = config.contact_damping
    contact_fric = config.contact_friction
    contact_fade = config.contact_fade_threshold  # smoothstep fade zone near separation

    @el.map
    def apply_forces(
        thrust: BodyThrust,
        drag: BodyDrag,
        pos: el.WorldPos,
        vel: el.WorldVel,
        inertia: el.Inertia,
        force: el.Force,
        sim_time: SimTime,
        motor_thrust: MotorThrust,
        noise_state: WakeNoise,
    ) -> tuple[el.Force, WakeNoise]:
        """Apply all forces to the body."""
        # Rotate body thrust to world frame
        quat = pos.angular()
        world_thrust = quat @ thrust

        # Gravity force
        gravity_force = el.SpatialForce(linear=gravity_vec * inertia.mass())

        # Linear drag
        drag_force = el.SpatialForce(linear=drag)

        # Angular drag (damping on rotation)
        omega = vel.angular()
        omega_mag = jnp.linalg.norm(omega)
        angular_drag_torque = -angular_drag * omega_mag * omega
        angular_drag_force = el.SpatialForce(torque=angular_drag_torque)

        # --- Multi-point ground contact ---
        # For each of the 4 contact points: transform to world, check penetration,
        # apply spring-damper normal force + lateral friction.
        # Forces are computed in world frame, torques from body-frame lever arms.
        com_pos = pos.linear()
        com_vel = vel.linear()
        omega_world = vel.angular()

        total_contact_force = jnp.zeros(3)
        total_contact_torque = jnp.zeros(3)

        def contact_point_force(i, carry):
            """Compute contact force for one point and accumulate."""
            acc_force, acc_torque = carry

            # Transform contact point from body to world frame
            pt_body = contact_pts_body[i]
            pt_world = com_pos + quat @ pt_body

            # Velocity at contact point: v_com + omega × r (world frame)
            r_world = quat @ pt_body  # offset from CoM in world frame
            pt_vel = com_vel + jnp.cross(omega_world, r_world)

            # Penetration depth (positive when below ground)
            penetration = ground_level - pt_world[2]

            # Graduated contact: smoothstep fade over contact_fade zone near separation
            # Prevents abrupt force discontinuity that causes liftoff gyro spike
            # smoothstep(x) = 3x² - 2x³ for x in [0,1]
            fade_t = jnp.clip(penetration / contact_fade, 0.0, 1.0)
            contact_fraction = fade_t * fade_t * (3.0 - 2.0 * fade_t)  # smoothstep

            # Normal force: spring + damper, clamped non-negative (unilateral)
            f_spring = contact_k * penetration
            f_damper = -contact_c * pt_vel[2]  # damp vertical velocity at point
            f_normal = jnp.maximum(f_spring + f_damper, 0.0)
            f_normal = f_normal * contact_fraction

            # Lateral friction: viscous damping of horizontal velocity at point
            f_friction_x = -contact_fric * pt_vel[0] * contact_fraction
            f_friction_y = -contact_fric * pt_vel[1] * contact_fraction

            # Total force at this contact point (world frame)
            pt_force_world = jnp.array([f_friction_x, f_friction_y, f_normal])

            # Torque from off-center force: tau = r × F (world frame)
            pt_torque_world = jnp.cross(r_world, pt_force_world)

            return (acc_force + pt_force_world, acc_torque + pt_torque_world)

        total_contact_force, total_contact_torque = jax.lax.fori_loop(
            0, 4, contact_point_force, (total_contact_force, total_contact_torque)
        )

        # Contact forces and torques are in world frame (matching el.SpatialForce convention)
        ground_contact_force = el.SpatialForce(
            linear=total_contact_force,
            torque=total_contact_torque,
        )

        # --- Wake turbulence: OU correlated noise ---
        # Two intensity components:
        # 1. Ground proximity (existing): propwash bounces off ground
        # 2. Descent-through-wake (new): flying into own downwash

        agl = pos.linear()[2] - ground_level
        proximity = jnp.clip((ge_height - agl) / ge_height, 0.0, 1.0)
        total_thrust_mag = jnp.sum(motor_thrust)
        thrust_fraction = jnp.clip(total_thrust_mag / (inertia.mass() * 9.81), 0.0, 1.0)

        # Ground proximity intensity (existing behavior, improved noise model)
        ground_intensity = proximity * proximity * thrust_fraction

        # Descent-through-wake intensity (new) — uses body-frame descent, consistent with VRS
        quat_inv = pos.angular().inverse()
        vel_body = quat_inv @ vel.linear()
        descent_body = jnp.maximum(-vel_body[2], 0.0)  # positive when descending in body frame
        v_induced = jnp.sqrt(jnp.maximum(total_thrust_mag, 0.01) / (2.0 * air_density * 4.0 * disk_area))
        descent_ratio = descent_body / jnp.maximum(v_induced, 0.1)
        descent_intensity = jnp.clip(descent_ratio, 0.0, 2.0) ** 2 * thrust_fraction

        # OU noise update: noise_next = noise_prev * decay + diffusion * randn
        # Use JAX deterministic PRNG keyed from tick for reproducibility
        tick_int = jnp.int32(jnp.round(sim_time[0] / dt))
        rng_key = jax.random.PRNGKey(tick_int)
        white_noise = jax.random.normal(rng_key, shape=(6,))
        new_noise = noise_state * ou_decay + ou_diffusion * white_noise

        # Scale noise by per-axis std dev and combined intensity
        force_std = jnp.array([ge_force_std, ge_force_std, ge_force_std])
        torque_std = jnp.array([ge_torque_std, ge_torque_std, ge_torque_std])

        # Add descent-specific turbulence on top of ground turbulence
        descent_force_std = jnp.array([pw_descent_force_std, pw_descent_force_std, pw_descent_force_std])
        descent_torque_std = jnp.array([pw_descent_torque_std, pw_descent_torque_std, pw_descent_torque_std])

        total_force_std = force_std * ground_intensity + descent_force_std * descent_intensity
        total_torque_std = torque_std * ground_intensity + descent_torque_std * descent_intensity

        wake_force = new_noise[:3] * total_force_std
        wake_torque = new_noise[3:] * total_torque_std

        wake_turbulence = el.SpatialForce(
            linear=wake_force,
            torque=wake_torque,
        )

        # Sum all forces
        total = force + world_thrust + gravity_force + drag_force + angular_drag_force + ground_contact_force + wake_turbulence
        return total, new_noise

    return apply_forces


def create_ground_constraint_system(config: DroneConfig):
    """
    Minimal anti-tunneling safety clamp.

    The primary ground stabilization is now handled by the multi-point contact
    forces in apply_forces. This post-integration clamp is only a safety net
    to prevent the center of mass from tunneling through the ground plane
    due to numerical integration overshoot.

    No angular damping — attitude stabilization on ground comes from the
    distributed contact forces creating natural restoring torques.
    """
    ground_level = config.ground_level

    @el.map
    def ground_constraint(pos: el.WorldPos, vel: el.WorldVel) -> tuple[el.WorldPos, el.WorldVel]:
        """Anti-tunneling clamp: prevent CoM from going below ground."""
        p = pos.linear()
        v = vel.linear()

        # Clamp position to ground level
        below_ground = p[2] < ground_level
        new_z = jnp.where(below_ground, ground_level, p[2])
        new_vz = jnp.where(below_ground & (v[2] < 0), 0.0, v[2])

        new_pos = el.SpatialTransform(
            linear=jnp.array([p[0], p[1], new_z]),
            angular=pos.angular(),
        )
        new_vel = el.SpatialMotion(
            linear=jnp.array([v[0], v[1], new_vz]),
            angular=vel.angular(),  # no angular modification
        )

        return new_pos, new_vel

    return ground_constraint


def create_time_update_system(config: DroneConfig):
    """Create system to track simulation time."""
    dt = config.sim_time_step

    @el.map
    def update_time(t: SimTime) -> SimTime:
        """Increment simulation time."""
        return t + dt

    return update_time


# --- World and System Construction ---


def create_world(config: DroneConfig) -> tuple[el.World, el.EntityId]:
    """
    Create the simulation world with a drone entity.

    Args:
        config: Drone configuration

    Returns:
        Tuple of (world, drone_entity_id)
    """
    world = el.World()

    # Initial state from config
    initial_pos = el.SpatialTransform(
        linear=jnp.array(config.initial_position),
        angular=el.Quaternion(jnp.array(config.initial_quaternion)),
    )
    initial_vel = el.SpatialMotion(
        linear=jnp.array(config.initial_velocity),
        angular=jnp.array(config.initial_angular_velocity),
    )
    inertia = el.SpatialInertia(
        mass=config.mass,
        inertia=jnp.array(config.inertia_diagonal),
    )

    # Spawn drone entity
    drone = world.spawn(
        [
            el.Body(
                world_pos=initial_pos,
                world_vel=initial_vel,
                inertia=inertia,
            ),
            Drone(sim_time=jnp.array([0.0])),
        ],
        name="drone",
    )

    return world, drone


def create_physics_system(config: DroneConfig) -> el.System:
    """
    Create the complete physics system for the simulation.

    Args:
        config: Drone configuration

    Returns:
        Combined physics system
    """
    # Create individual systems
    motor_dynamics = create_motor_dynamics(config)
    rotor_aero = create_rotor_aero_system(config)
    body_thrust = create_body_thrust_system(config)
    drag = create_drag_system(config)
    apply_forces = create_apply_forces_system(config)
    ground = create_ground_constraint_system(config)
    time_update = create_time_update_system(config)

    # Effector systems (applied before integration)
    # Pipeline: motor_dynamics → rotor_aero (IGE/VRS) → body_thrust → drag → apply_forces
    effectors = motor_dynamics | rotor_aero | body_thrust | drag | apply_forces

    # 6-DOF integrator with effectors
    physics = el.six_dof(
        config.sim_time_step,
        effectors,
        integrator=el.Integrator.SemiImplicit,
    )

    # Post-integration systems
    post_systems = ground | time_update

    return physics | post_systems


# --- Helper Functions ---

# Note: For SITL integration, we use the world.run() callback mechanism
# rather than direct component access. See main.py for integration example.


if __name__ == "__main__":
    # Quick test of the physics compilation
    from config import DEFAULT_CONFIG
    import polars as pl

    config = DEFAULT_CONFIG
    config.simulation_time = 0.2  # Short test
    config.set_as_global()

    print("Testing physics simulation...")
    print(f"Config: {config.mass}kg, dt={config.sim_time_step * 1000}ms")
    print(f"Hover throttle: {config.hover_throttle:.1%}")

    # Test 1: Free fall (motors off)
    print("\n--- Free fall test (motors off) ---")
    world, drone = create_world(config)
    system = create_physics_system(config)
    exec = world.build(system)

    # Run 200 ticks (200ms at 1kHz)
    exec.run(200)

    # Get history data
    df = exec.history(["drone.world_pos", "drone.world_vel", "drone.sim_time"])
    print(f"Simulated {len(df)} ticks")

    # world_pos is [qx, qy, qz, qw, x, y, z] (Elodin scalar-last format) - extract using polars
    df_expanded = df.select(
        pl.col("drone.world_pos").arr.get(6).alias("z"),
        pl.col("drone.world_vel").arr.get(2).alias("vz"),
    )

    final_z = df_expanded[-1]["z"]
    final_vz = df_expanded[-1]["vz"]

    print(f"Final z: {final_z} m (started at {config.initial_position[2]:.3f} m)")
    print(f"Final vz: {final_vz} m/s")
    print(f"Expected fall: {0.5 * 9.81 * 0.2**2:.3f} m")

    # Check if z changed (indicates gravity is working)
    initial_z = df_expanded[0]["z"]
    delta_z = initial_z - final_z
    print(f"Actual fall: {delta_z} m")

    # Test 2: Check physics system builds correctly
    print("\n--- System compilation test ---")
    print("All physics systems compiled successfully!")

    print("\nPhysics test complete!")
