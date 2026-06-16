import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Lite 6 URDF joint frames: [xyz_offset, rpy_offset] per joint
JOINTS = [
    [[0,       0,       0.2435], [0,        0,        0       ]],
    [[0,       0,       0     ], [np.pi/2, -np.pi/2,  np.pi   ]],
    [[0.2002,  0,       0     ], [-np.pi,   0,        np.pi/2 ]],
    [[0.087,  -0.22761, 0     ], [np.pi/2,  0,        0       ]],
    [[0,       0,       0     ], [np.pi/2,  0,        0       ]],
    [[0,       0.0625,  0     ], [-np.pi/2, 0,        0       ]],
]

JOINT_NAMES    = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
JOINT_MIN      = np.array([-6.28, -2.61, -0.06, -6.28, -2.16, -6.28])
JOINT_MAX      = np.array([ 6.28,  2.61,  5.23,  6.28,  2.16,  6.28])
INIT_THETA     = np.array([0.0, 0.7, 1.57, 0.0, 0.87, 0.0])
TARGET_RPY     = [np.pi, 0.0, 0.0]

IK_DAMPING     = 0.1
IK_ALPHA       = 0.5
IK_MAX_STEPS   = 200
IK_THRESHOLD   = 1e-3
IK_STEP_CLIP   = 0.1
IK_JUMP_LIMIT  = 1.5


def euler_rotation_matrix(x, y, z):
    Rx = np.array([[1,         0,          0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x),  np.cos(x)]])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [ 0,         1,         0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z),  np.cos(z), 0],
                   [0,          0,         1]])
    return Rz @ Ry @ Rx


def make_transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = euler_rotation_matrix(*rpy)
    T[:3,  3] = xyz
    return T


def rotation_to_quaternion(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def FK(theta):
    """Forward kinematics. Returns (T_ee, pos, quat, z_axes, joint_origins)."""
    z_axes       = []
    joint_origins = []

    T = np.eye(4)
    for i, joint in enumerate(JOINTS):
        T = T @ make_transform(joint[0], joint[1])
        z_axes.append(T[:3, 2])
        joint_origins.append(T[:3, 3])
        T = T @ make_transform([0, 0, 0], [0, 0, theta[i]])

    pos  = T[:3, 3]
    quat = rotation_to_quaternion(T[:3, :3])
    return T, pos, quat, z_axes, joint_origins


class IK_Calculate(Node):
    def __init__(self):
        super().__init__('IK_calculate')
        self.target_sub = self.create_subscription(Point, '/target_pos', self._cb_target_pos, 10)
        self.joint_pub  = self.create_publisher(JointTrajectory, '/lite6_traj_controller/joint_trajectory', 10)
        self.theta = INIT_THETA.copy()

    def _cb_target_pos(self, msg: Point):
        dist = np.sqrt(msg.x**2 + msg.y**2 + msg.z**2)
        if (dist <= 0.5 and msg.z <= 0.05) or msg.z < 0:
            self.get_logger().warn(f"denied: pos=[{msg.x:.3f},{msg.y:.3f},{msg.z:.3f}]")
            return

        target_pos = np.array([msg.x, msg.y, msg.z])
        target_R   = euler_rotation_matrix(*TARGET_RPY)

        theta = self.theta.copy()
        converged = False
        pos = np.zeros(3)

        for step in range(IK_MAX_STEPS):
            T, pos, _, z_axes, joint_origins = FK(theta)

            R_error = target_R @ T[:3, :3].T
            dP = target_pos - pos
            dW = np.array([R_error[2, 1] - R_error[1, 2],
                           R_error[0, 2] - R_error[2, 0],
                           R_error[1, 0] - R_error[0, 1]]) * 0.5
            dX = np.append(dP, dW)

            if np.linalg.norm(dX) < IK_THRESHOLD:
                self.get_logger().info(f"[Success] Converged at step {step}")
                converged = True
                break

            J = np.zeros((6, 6))
            for i in range(6):
                r = pos - joint_origins[i]
                J[:3, i] = np.cross(z_axes[i], r)
                J[3:, i] = z_axes[i]

            J_pinv  = J.T @ np.linalg.inv(J @ J.T + IK_DAMPING**2 * np.eye(6))
            d_theta = np.clip(J_pinv @ dX, -IK_STEP_CLIP, IK_STEP_CLIP)
            theta   = theta + IK_ALPHA * d_theta
            theta   = np.arctan2(np.sin(theta), np.cos(theta))
            theta   = np.clip(theta, JOINT_MIN, JOINT_MAX)

        if not converged:
            self.get_logger().warn(f"IK failed to converge for [{msg.x:.3f},{msg.y:.3f},{msg.z:.3f}]")
            return

        delta = np.max(np.abs(theta - self.theta))
        if delta > IK_JUMP_LIMIT:
            self.get_logger().warn(f"IK jump too large: {delta:.3f} rad, skipping")
            return

        self.theta = theta
        move_sec   = max(3.0, float(delta) * 8.0)

        pt = JointTrajectoryPoint()
        pt.positions       = self.theta.tolist()
        pt.time_from_start = Duration(sec=int(move_sec), nanosec=int((move_sec % 1) * 1e9))

        joint_msg = JointTrajectory()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.joint_names  = JOINT_NAMES
        joint_msg.points       = [pt]

        self.joint_pub.publish(joint_msg)
        self.get_logger().info(f"Target: [{msg.x:.3f},{msg.y:.3f},{msg.z:.3f}] -> EE: {pos}")


def main(args=None):
    rclpy.init(args=args)
    node = IK_Calculate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
