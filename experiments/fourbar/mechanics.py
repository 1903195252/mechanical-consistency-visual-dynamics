"""Independent planar rigid-body mechanics. This module does not import MuJoCo."""

import numpy as np

# Absolute link angles q=(crank, coupler, rocker), MuJoCo tree angles u.
TREE_TO_ABSOLUTE = np.array([[1., 0., 0.], [1., 1., 0.], [0., 0., 1.]])
# Relative angles at the four real joints A, B, C, D.
JOINT_RATE_MAP = np.array([[1., 0., 0.], [-1., 1., 0.], [0., -1., 1.], [0., 0., 1.]])


def direction(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def tangent(theta):
    return np.array([-np.sin(theta), np.cos(theta)])


def properties(config):
    lengths = np.asarray(config['lengths_m'])
    height, depth = config['bar_height_m'], config['bar_depth_m']
    # Capsule = central rectangle + two semicircular ends, minus two bores.
    # L is pivot spacing; overall length is L + 2R. Symmetry fixes COM at L/2.
    radius, bore = height/2, config['joint_bore_radius_m']
    if not 0 < bore < radius or np.any(lengths <= 2*radius):
        raise ValueError('Invalid drilled link dimensions')
    area = 2*radius*lengths + np.pi*radius**2 - 2*np.pi*bore**2
    mass_per_area = depth*config['density_kg_m3']
    masses = area*mass_per_area
    x_squared = (2*radius*lengths**3/12 + np.pi*radius**2*lengths**2/4
                 + 4*lengths*radius**3/3 + np.pi*radius**4/4
                 - np.pi*bore**2*lengths**2/2 - np.pi*bore**4/2)
    z_squared = 2*lengths*radius**3/3 + np.pi*radius**4/4 - np.pi*bore**4/2
    inertias = np.stack([
        mass_per_area*z_squared + masses*depth**2/12,
        mass_per_area*(x_squared+z_squared),
        mass_per_area*x_squared + masses*depth**2/12,
    ], axis=1)
    return masses, inertias


def payload_inertia(config):
    """Uniform rectangular payload about its COM; local X/Y/Z match the coupler."""
    size=np.asarray(config['payload_size_m'],dtype=float)
    if config['payload_shape']!='cuboid' or size.shape!=(3,) or np.any(size<=0):
        raise ValueError('Payload must be a cuboid with three positive dimensions')
    if config['payload_mass_kg']<=0:
        raise ValueError('Payload mass must be positive')
    return config['payload_mass_kg']*(np.sum(size**2)-size**2)/12


def inverse_kinematics(crank_angle, config):
    """Circle intersection chooses a fixed assembly branch, only for initialization/reference."""
    l1, l2, l3 = config['lengths_m']
    b = l1 * direction(crank_angle)
    d = np.array([config['ground_length_m'], 0.])
    separation = d - b
    distance = np.linalg.norm(separation)
    unit = separation / distance
    along = (l2*l2 - l3*l3 + distance*distance) / (2*distance)
    height_squared = l2*l2 - along*along
    if height_squared <= 0:
        raise ValueError('Unreachable or singular four-bar configuration')
    normal = np.array([-unit[1], unit[0]])
    c = b + along*unit + config['assembly_branch'] * np.sqrt(height_squared)*normal
    return np.array([crank_angle, np.arctan2(*(c-b)[::-1]), np.arctan2(*(c-d)[::-1])])


def closure(q, config):
    l1, l2, l3 = config['lengths_m']
    return l1*direction(q[0]) + l2*direction(q[1]) - l3*direction(q[2]) - np.array([config['ground_length_m'], 0.])


def closure_jacobian(q, config):
    l1, l2, l3 = config['lengths_m']
    return np.column_stack([l1*tangent(q[0]), l2*tangent(q[1]), -l3*tangent(q[2])])


def closure_curvature(q, velocity, config):
    l1, l2, l3 = config['lengths_m']
    return -l1*direction(q[0])*velocity[0]**2 - l2*direction(q[1])*velocity[1]**2 + l3*direction(q[2])*velocity[2]**2


def coupler_point(q, config, along, normal_offset=0.):
    l1 = config['lengths_m'][0]
    position = l1*direction(q[0]) + along*direction(q[1]) + normal_offset*tangent(q[1])
    jacobian = np.column_stack([l1*tangent(q[0]), along*tangent(q[1])-normal_offset*direction(q[1]), np.zeros(2)])
    return position, jacobian


def mass_bias_energy(q, velocity, config):
    """M, centrifugal/Coriolis c, gravity G, kinetic/potential energies from COM Jacobians."""
    l1, l2, l3 = config['lengths_m']
    masses, inertias = properties(config)
    zero = np.zeros(2)
    positions = [l1/2*direction(q[0]), l1*direction(q[0])+l2/2*direction(q[1]),
                 np.array([config['ground_length_m'], 0.])+l3/2*direction(q[2])]
    jacobians = [np.column_stack([l1/2*tangent(q[0]), zero, zero]),
                 np.column_stack([l1*tangent(q[0]), l2/2*tangent(q[1]), zero]),
                 np.column_stack([zero, zero, l3/2*tangent(q[2])])]
    curvature = [-l1/2*direction(q[0])*velocity[0]**2,
                 -l1*direction(q[0])*velocity[0]**2-l2/2*direction(q[1])*velocity[1]**2,
                 -l3/2*direction(q[2])*velocity[2]**2]
    payload = config['payload_mass_kg']
    along, normal = config['payload_along_coupler_m'], config['payload_normal_offset_m']
    position, jacobian = coupler_point(q, config, along, normal)
    positions.append(position); jacobians.append(jacobian)
    curvature.append(-l1*direction(q[0])*velocity[0]**2-(along*direction(q[1])+normal*tangent(q[1]))*velocity[1]**2)
    masses = np.append(masses, payload)
    rotational_inertia = inertias[:, 1].copy()
    rotational_inertia[0] += config['motor_armature_kg_m2']
    rotational_inertia[1] += payload_inertia(config)[1]
    mass_matrix = np.diag(rotational_inertia)
    coriolis, gravity = np.zeros(3), np.zeros(3)
    potential = 0.
    g = np.array([0., -config['gravity_m_s2']])
    for mass, position, jacobian, centripetal in zip(masses, positions, jacobians, curvature):
        mass_matrix += mass * jacobian.T @ jacobian
        coriolis += mass * jacobian.T @ centripetal
        gravity -= mass * jacobian.T @ g
        potential += mass*config['gravity_m_s2']*(position[1]+config['pivot_height_m'])
    kinetic = 0.5 * velocity @ mass_matrix @ velocity
    return mass_matrix, coriolis, gravity, kinetic, potential


def friction(velocity, config):
    rates = JOINT_RATE_MAP @ velocity
    b = np.asarray(config['viscous_friction_Nm_s_rad'])
    c = np.asarray(config['coulomb_friction_Nm'])
    opposing_torques = b*rates + c*np.tanh(rates/config['coulomb_smoothing_rad_s'])
    generalized = -JOINT_RATE_MAP.T @ opposing_torques
    return generalized, rates, -opposing_torques


def drive_torque(t, crank_velocity, config):
    drive = config['drive']
    if drive['type']=='open_loop_multisine_torque':
        command = drive['torque_bias_Nm'] + drive['torque_modulation_Nm']*np.sin(2*np.pi*drive['modulation_hz']*t)
        command += drive['excitation_Nm']*np.sin(2*np.pi*drive['excitation_hz']*t)
    elif drive['type']=='bounded_velocity_feedback_to_torque':
        speed = drive['speed_rad_s'] + drive['speed_modulation_rad_s']*np.sin(2*np.pi*drive['modulation_hz']*t)
        command = drive['gain_Nm_s_rad']*(speed-crank_velocity) + drive['excitation_Nm']*np.sin(2*np.pi*drive['excitation_hz']*t)
    else:
        raise ValueError('Unknown torque input mode')
    return float(np.clip(command, -drive['torque_limit_Nm'], drive['torque_limit_Nm']))


def reduced_acceleration(crank_angle, crank_velocity, torque, config):
    """Independent ideal-constraint 1-DOF equation; no simulator mass/force queries."""
    q = inverse_kinematics(crank_angle, config)
    jacobian = closure_jacobian(q, config)
    derivatives = np.r_[1., np.linalg.solve(jacobian[:, 1:], -jacobian[:, 0])]
    velocity = derivatives*crank_velocity
    curvature = np.r_[0., np.linalg.solve(jacobian[:, 1:], -closure_curvature(q, velocity, config))]
    mass, coriolis, gravity, _, _ = mass_bias_energy(q, velocity, config)
    friction_force, _, _ = friction(velocity, config)
    _, point_jacobian = coupler_point(q, config, config['force_application_along_coupler_m'])
    external = point_jacobian.T @ np.asarray(config['external_force_xz_N'])
    applied = np.array([torque, 0., 0.])+friction_force+external
    return float(derivatives @ (applied-coriolis-gravity-mass@curvature) / (derivatives@mass@derivatives))
