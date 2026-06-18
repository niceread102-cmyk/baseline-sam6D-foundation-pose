"""Arm and gripper control abstractions.

XArmController connects directly to the Lite 6 via the xArm Python SDK (TCP).
No ROS 2 stack required.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .ik_solver import IKSolver

log = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class ArmController(ABC):
    @property
    @abstractmethod
    def current_theta(self) -> np.ndarray:
        """Current joint angles in radians."""

    @abstractmethod
    def move_to_xyz(self, xyz: np.ndarray, wait: bool = True) -> bool:
        """Move end-effector to Cartesian target (metres, robot-base frame)."""

    @abstractmethod
    def move_to_angles(self, theta: np.ndarray, wait: bool = True, safe: bool = True) -> bool:
        """Move to explicit joint angles (radians).

        safe=False bypasses the per-joint delta safety check — only use for
        recovery homing moves where the arm may be in an unexpected position.
        """

    @abstractmethod
    def set_gripper(self, open: bool) -> None:
        """Open or close the gripper."""

    def recover(self) -> None:
        """Re-initialize the arm after an error or stop state."""


# ── xArm SDK implementation ───────────────────────────────────────────────────

class XArmController(ArmController):
    """Controls the Lite 6 directly via the xArm Python SDK (no ROS required)."""

    def __init__(
        self,
        robot_ip: str = "192.168.1.150",
        move_speed: float = 0.3,   # rad/s
        move_acc: float = 5.0,     # rad/s²
        max_joint_delta: float = 1.5,  # rad; safety limit per joint per command (0 = disabled)
        ik_solver: IKSolver | None = None,
    ):
        from xarm import XArmAPI

        self._arm = XArmAPI(port=robot_ip, is_radian=True)
        self._arm.connect()
        if not self._arm.connected:
            raise RuntimeError(f"Failed to connect to xArm at {robot_ip}")

        self._arm.clean_error()
        self._arm.clean_warn()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)   # position control
        self._arm.set_state(0)  # motion state

        self._ik              = ik_solver
        self._speed           = move_speed
        self._acc             = move_acc
        self._max_joint_delta = max_joint_delta
        self._poll_interval   = 0.25  # seconds between in-motion EE position log lines

        log.info("Connected to xArm at %s", robot_ip)

        # FK sanity check: compare robot controller's FK with our Python model
        if ik_solver is not None:
            try:
                code2, theta_now = self._arm.get_servo_angle(is_radian=True)
                code1, pos_now   = self._arm.get_position(is_radian=True)
                if code1 == 0 and code2 == 0:
                    fk_pos, _ = ik_solver.fk(np.array(theta_now[:6]))
                    robot_pos = [pos_now[0] / 1000, pos_now[1] / 1000, pos_now[2] / 1000]
                    log.info(
                        "FK check — robot=[%.3f, %.3f, %.3f]m  "
                        "our_FK=[%.3f, %.3f, %.3f]m  "
                        "delta=[%.3f, %.3f, %.3f]m",
                        robot_pos[0], robot_pos[1], robot_pos[2],
                        fk_pos[0], fk_pos[1], fk_pos[2],
                        robot_pos[0] - fk_pos[0],
                        robot_pos[1] - fk_pos[1],
                        robot_pos[2] - fk_pos[2],
                    )
            except Exception as e:
                log.debug("FK check skipped: %s", e)

    # ── state ─────────────────────────────────────────────────────────────

    @property
    def current_theta(self) -> np.ndarray:
        code, angles = self._arm.get_servo_angle(is_radian=True)
        if code != 0:
            log.warning("get_servo_angle returned code=%d", code)
            return np.zeros(6)
        return np.array(angles[:6])

    # ── movement ──────────────────────────────────────────────────────────

    def _poll_position_loop(self, stop_event: threading.Event) -> None:
        """Background thread: log EE position every poll_interval seconds during a move."""
        while not stop_event.wait(self._poll_interval):
            code, theta = self._arm.get_servo_angle(is_radian=True)
            if code == 0 and self._ik is not None:
                fk_pos, _ = self._ik.fk(np.array(theta[:6]))
                log.info(
                    "  ↳ EE=[%.3f, %.3f, %.3f]m  joints=%s",
                    fk_pos[0], fk_pos[1], fk_pos[2],
                    np.round(np.array(theta[:6]), 3).tolist(),
                )

    def move_to_angles(self, theta: np.ndarray, wait: bool = True, safe: bool = True) -> bool:
        theta = np.asarray(theta, dtype=float)

        # Safety: reject moves where any joint changes by more than max_joint_delta
        if safe and self._max_joint_delta > 0:
            current = self.current_theta
            delta   = np.abs(theta - current)
            if np.any(delta > self._max_joint_delta):
                log.error(
                    "SAFETY REJECT: joint delta %s rad exceeds max_joint_delta=%.2f — "
                    "IK solution may be wrong (check p_obj_base log and T_cam_base)",
                    np.round(delta, 3).tolist(), self._max_joint_delta,
                )
                return False

        log.info("→ CMD angles %s", np.round(theta, 3).tolist())

        # Start background position polling so EE trajectory is visible in logs
        stop_poll = threading.Event()
        poll_thread: threading.Thread | None = None
        if wait and self._ik is not None:
            poll_thread = threading.Thread(
                target=self._poll_position_loop, args=(stop_poll,),
                daemon=True, name="ArmPoller",
            )
            poll_thread.start()

        try:
            code = self._arm.set_servo_angle(
                angle=theta.tolist(),
                speed=self._speed,
                mvacc=self._acc,
                wait=wait,
                is_radian=True,
            )
        finally:
            stop_poll.set()
            if poll_thread is not None:
                poll_thread.join(timeout=1.0)

        if code not in (0, None):
            log.error("set_servo_angle failed (code=%d)", code)
            if self._arm.state in (4, 5):
                log.warning("Arm in stop/error state — auto-recovering")
                self._recover()
            return False

        # Log actual post-move joint angles and FK end-effector position
        if wait:
            code2, actual = self._arm.get_servo_angle(is_radian=True)
            if code2 == 0:
                actual_arr = np.array(actual[:6])
                if self._ik is not None:
                    fk_pos, _ = self._ik.fk(actual_arr)
                    log.info(
                        "✓ ACT angles %s  EE=[%.3f, %.3f, %.3f]m",
                        np.round(actual_arr, 3).tolist(),
                        fk_pos[0], fk_pos[1], fk_pos[2],
                    )
                else:
                    log.info("✓ ACT angles %s", np.round(actual_arr, 3).tolist())

        return True

    def move_to_xyz(self, xyz: np.ndarray, wait: bool = True) -> bool:
        if self._ik is None:
            raise RuntimeError("XArmController requires ik_solver for move_to_xyz")
        xyz = np.asarray(xyz, dtype=float)
        theta_sol, converged = self._ik.solve(xyz, self.current_theta)
        if not converged or theta_sol is None:
            log.error("IK failed to converge for xyz=%s", xyz.tolist())
            return False
        log.info("→ xyz %s  (IK ok)", np.round(xyz, 3).tolist())
        return self.move_to_angles(theta_sol, wait=wait)

    # ── gripper ───────────────────────────────────────────────────────────

    def set_gripper(self, open: bool) -> None:
        if open:
            code = self._arm.open_lite6_gripper()
            log.info("Gripper: open (code=%s)", code)
        else:
            code = self._arm.close_lite6_gripper()
            log.info("Gripper: close (code=%s)", code)
        time.sleep(0.5)

    # ── recovery ──────────────────────────────────────────────────────────

    def _recover(self) -> None:
        self._arm.clean_error()
        self._arm.clean_warn()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(0.3)
        # Reseat the commanded position to the actual position so the arm
        # does not spring to a stale commanded target after fault clearance.
        code, angles = self._arm.get_servo_angle(is_radian=True)
        if code == 0 and angles:
            self._arm.set_servo_angle(
                angle=angles[:6],
                speed=0.05,
                mvacc=1.0,
                wait=True,
                is_radian=True,
            )
        log.info("Arm recovery complete")

    def recover(self) -> None:
        self._recover()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._arm.disconnect()
