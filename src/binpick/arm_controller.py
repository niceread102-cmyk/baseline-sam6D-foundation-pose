"""Robot arm and gripper control abstractions.

``XArmController`` connects directly to the Lite 6 via the xArm Python SDK
(TCP socket).  No ROS 2 stack is required.

Thread model
------------
``move_to_angles(wait=True)`` spawns a short-lived daemon thread (``ArmPoller``)
that logs the EE position every ``_POLL_INTERVAL_S`` seconds while the arm is
in motion.  The move call blocks until the arm reports completion, then the
thread is stopped before returning.

Move verification
------------------
The xArm SDK's ``set_servo_angle(wait=True)`` returning success only means its
internal ``wait_move()`` observed the motion *state* clear — not that the arm
reached the commanded joints. A collision/torque-limit stop mid-move, or a
race in ``wait_move()``'s state polling when two moves are issued back-to-back
with no gap, can both report success while the arm silently stalls partway.
``move_to_angles`` therefore reads back the actual joint angles after every
``wait=True`` move and treats a mismatch beyond ``_POSITION_VERIFY_TOL_RAD`` as
a failure — this is what catches a partial/aborted move *before* the caller
(e.g. ``BinPicker._execute_pick``) proceeds to close the gripper on it. A
``_MOVE_SETTLE_S`` pause after every confirmed move also guards against the
back-to-back race itself.

Unit conventions
----------------
- Joint angles: **radians** (xArm SDK is initialised with ``is_radian=True``).
- Cartesian positions exposed by this module: **metres**.
- The xArm SDK's own FK (``get_position``) returns positions in **millimetres**;
  these are converted by ``_XARM_MM_TO_M`` before any logging or comparison.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .ik_solver import IKSolver

log = logging.getLogger(__name__)

# ── Timing and motion constants ───────────────────────────────────────────────
_POLL_INTERVAL_S    = 0.25   # seconds between EE position log lines during a move
_GRIPPER_SETTLE_S   = 0.5    # wait after close_gripper before moving away
_RELEASE_SETTLE_S   = 0.3    # wait after open_gripper (release) before moving
_RECOVERY_SETTLE_S  = 0.3    # wait after fault-clear before reseating commanded position
_RECOVERY_SPEED     = 0.05   # rad/s — very slow reseat move after fault recovery
_RECOVERY_ACC       = 1.0    # rad/s²
_XARM_MM_TO_M       = 1e-3   # xArm FK returns mm; multiply to get metres

# Post-move verification: the xArm SDK's set_servo_angle(wait=True) can report
# success (code=0) even when the arm never reached the commanded target — e.g.
# a collision/torque-limit stop partway through, or a race in the SDK's own
# wait_move() (which polls motion *state*, not final position, so a command
# issued with ~0 gap after the previous move can observe a stale "not moving"
# reading and return early). Verify actual vs. commanded joints after every
# wait=True move and treat a mismatch as a failure instead of trusting the SDK.
_POSITION_VERIFY_TOL_RAD = 0.05   # rad (~2.9°) — per-joint tolerance for "did we get there"
_MOVE_SETTLE_S           = 0.1    # seconds after a confirmed move, before the next command


# ── Abstract base ─────────────────────────────────────────────────────────────

class ArmController(ABC):
    """Interface for robot arm and gripper control.

    All positions are in **metres** (robot-base frame) and all angles are in
    **radians**.  Implementations must be safe to call from the main thread
    while a background preview thread may be reading sensor state.
    """

    @property
    @abstractmethod
    def current_theta(self) -> np.ndarray:
        """Current joint angles in radians, shape (6,)."""

    @abstractmethod
    def move_to_xyz(self, xyz: np.ndarray, wait: bool = True) -> bool:
        """Move end-effector to a Cartesian target (metres, robot-base frame).

        Returns True on success, False if IK failed or the arm rejected the move.
        """

    @abstractmethod
    def move_to_angles(self, theta: np.ndarray, wait: bool = True, safe: bool = True) -> bool:
        """Move to explicit joint angles (radians).

        Parameters
        ----------
        theta:
            Target joint configuration, shape (6,), in radians.
        wait:
            Block until the arm reports completion.
        safe:
            If True, reject moves where any joint changes by more than
            ``max_joint_delta`` from the current position.  Set False only for
            recovery homing moves where the arm may be in an unexpected position.

        Returns True on success, False if the move was rejected OR (when
        ``wait=True``) if the arm's actual final position doesn't match
        ``theta`` — callers rely on False meaning "did not get there," not just
        "the command was rejected," so implementations must verify the actual
        result rather than trusting a lower-level SDK's return code alone (see
        ``XArmController.move_to_angles`` for why: some SDKs report success
        even when a collision/fault stopped the arm short of the target).
        """

    @abstractmethod
    def set_gripper(self, open: bool) -> None:
        """Open or close the gripper.

        Blocks for a short settle period after issuing the command so the
        fingers reach their final position before the next arm motion.
        """

    def recover(self) -> None:
        """Re-initialise the arm after an error or stop state.

        Clears faults, re-enables motion, and reseats the commanded position to
        the actual position so the arm does not spring to a stale target after
        fault clearance.  Override in subclasses for hardware-specific recovery.
        """


# ── xArm SDK implementation ───────────────────────────────────────────────────

class XArmController(ArmController):
    """Controls the Lite 6 directly via the xArm Python SDK (no ROS required).

    On construction, connects to the arm, clears any existing faults, and
    optionally runs a one-time FK sanity check to compare the arm controller's
    built-in FK against the Python ``Lite6IKSolver`` model.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.150",
        move_speed: float = 0.3,    # rad/s
        move_acc: float = 5.0,      # rad/s²
        max_joint_delta: float = 1.5,  # rad; per-joint safety limit per command (0 = disabled)
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

        log.info("Connected to xArm at %s", robot_ip)

        # Sync the real, controller-configured TCP offset (flange -> gripper
        # tip, set once via scripts/calibrate_T_cam_base.py's --tcp-offset and
        # persisted on the controller) into our own Lite6IKSolver model, which
        # otherwise computes FK/IK for the bare flange. Without this, every
        # Cartesian target (hover, pick, place) is silently reached ~1 tool
        # length short along the approach axis — confirmed as the cause of a
        # real hardware collision (the arm ends up well past the intended
        # grasp point). Reading it from the arm itself (ground truth for
        # whatever gripper is physically attached) instead of hardcoding it in
        # config keeps the two in sync even if the arm is recalibrated.
        if ik_solver is not None and hasattr(ik_solver, "set_tcp_offset"):
            try:
                time.sleep(0.3)  # let the real-time report stream populate tcp_offset
                offset = self._arm.tcp_offset  # [x,y,z (mm), roll,pitch,yaw (rad)]
                xyz_m  = [v * _XARM_MM_TO_M for v in offset[:3]]
                rpy_rad = list(offset[3:6])
                ik_solver.set_tcp_offset(xyz_m, rpy_rad)
                log.info(
                    "Synced TCP offset from arm into IK solver: xyz=%s m  rpy=%s rad",
                    [round(v, 4) for v in xyz_m], [round(v, 4) for v in rpy_rad],
                )
            except Exception as e:
                log.warning(
                    "Could not read TCP offset from arm (%s) — IK solver will use "
                    "whatever tcp_offset it was constructed with (default: none). "
                    "Cartesian targets may be off by the flange-to-tip distance.", e,
                )

        # One-time FK sanity check: compare the robot controller's built-in FK
        # with our Python model so any URDF / firmware / TCP-offset mismatch is
        # caught early. Should now be small (a few mm) — a large, systematic
        # delta here (especially one that grows/shrinks with a fixed direction
        # relative to the EE) means the TCP offset sync above didn't take.
        if ik_solver is not None:
            try:
                code2, theta_now = self._arm.get_servo_angle(is_radian=True)
                code1, pos_now   = self._arm.get_position(is_radian=True)
                if code1 == 0 and code2 == 0:
                    fk_pos, _ = ik_solver.fk(np.array(theta_now[:6]))
                    robot_pos = np.array(pos_now[:3]) * _XARM_MM_TO_M
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
        """Daemon thread: log EE position at ``_POLL_INTERVAL_S`` intervals during a move.

        Runs until ``stop_event`` is set.  Requires an IK solver to be available;
        logs nothing if FK cannot be computed.
        """
        while not stop_event.wait(_POLL_INTERVAL_S):
            code, theta = self._arm.get_servo_angle(is_radian=True)
            if code == 0 and self._ik is not None:
                fk_pos, _ = self._ik.fk(np.array(theta[:6]))
                log.info(
                    "  ↳ EE=[%.3f, %.3f, %.3f]m  joints=%s",
                    fk_pos[0], fk_pos[1], fk_pos[2],
                    np.round(np.array(theta[:6]), 3).tolist(),
                )

    def _log_fault_reason(self) -> None:
        """Log the controller's actual error/warning code after a move failure.

        ``show=True`` also makes the xArm SDK itself log a human-readable
        description (error/warning title) via its own logger.
        """
        try:
            code, (err_code, warn_code) = self._arm.get_err_warn_code(show=True)
            if code == 0:
                log.error("Controller fault reason: error_code=%s warn_code=%s", err_code, warn_code)
        except Exception as e:
            log.debug("get_err_warn_code failed: %s", e)

    def move_to_angles(self, theta: np.ndarray, wait: bool = True, safe: bool = True) -> bool:
        """Move to explicit joint angles (radians).

        When ``safe=True``, the move is rejected if any joint would change by
        more than ``max_joint_delta`` from its current position.  This is a
        Python-level check; the xArm SDK applies its own independent safety
        limits in firmware regardless of this flag.
        """
        theta = np.asarray(theta, dtype=float)

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

        # Start background position polling so the EE trajectory is visible in logs
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
            self._log_fault_reason()
            if self._arm.state in (4, 5):
                log.warning("Arm in stop/error state — auto-recovering")
                self._recover()
            return False

        # Verify the arm actually reached the commanded target — set_servo_angle's
        # return code only reflects whether the SDK's wait_move() saw the motion
        # *state* clear, not whether the final position matches what we asked
        # for. A collision/torque stop mid-move, or a race in wait_move() when
        # commands are issued back-to-back, can both report success here while
        # the arm silently stalls partway. Catch that before the caller trusts
        # this move (e.g. before closing the gripper on a partial/wrong pose).
        if wait:
            code2, actual = self._arm.get_servo_angle(is_radian=True)
            if code2 != 0:
                log.error(
                    "Post-move verification failed — could not read back servo "
                    "angles (code=%d); treating move as failed.", code2,
                )
                return False

            actual_arr = np.array(actual[:6])
            verify_delta = np.abs(actual_arr - theta)
            if np.any(verify_delta > _POSITION_VERIFY_TOL_RAD):
                log.error(
                    "MOVE DID NOT REACH TARGET: commanded %s but arm is at %s "
                    "(delta=%s rad, tol=%.3f) — set_servo_angle reported success "
                    "but the joints never got there (likely a collision/torque "
                    "stop, or a wait_move() race from back-to-back commands).",
                    np.round(theta, 3).tolist(), np.round(actual_arr, 3).tolist(),
                    np.round(verify_delta, 3).tolist(), _POSITION_VERIFY_TOL_RAD,
                )
                self._log_fault_reason()
                if self._arm.state in (4, 5):
                    log.warning("Arm in stop/error state — auto-recovering")
                    self._recover()
                return False

            if self._ik is not None:
                fk_pos, _ = self._ik.fk(actual_arr)
                log.info(
                    "✓ ACT angles %s  EE=[%.3f, %.3f, %.3f]m",
                    np.round(actual_arr, 3).tolist(),
                    fk_pos[0], fk_pos[1], fk_pos[2],
                )
            else:
                log.info("✓ ACT angles %s", np.round(actual_arr, 3).tolist())

            # Settle gap before any immediately-following move command, to avoid
            # racing wait_move()'s motion-state polling on the next call.
            time.sleep(_MOVE_SETTLE_S)

        return True

    def move_to_xyz(self, xyz: np.ndarray, wait: bool = True) -> bool:
        """Move EE to a Cartesian target via IK.

        Requires an ``ik_solver`` to have been provided at construction.
        Returns False if IK fails to converge.
        """
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
        """Open or close the Lite 6 gripper.

        Blocks for ``_GRIPPER_SETTLE_S`` after issuing the command so the
        fingers physically reach their final position before the next arm motion.
        """
        if open:
            code = self._arm.open_lite6_gripper()
            log.info("Gripper: open (code=%s)", code)
        else:
            code = self._arm.close_lite6_gripper()
            log.info("Gripper: close (code=%s)", code)
        time.sleep(_GRIPPER_SETTLE_S)

    # ── recovery ──────────────────────────────────────────────────────────

    def _recover(self) -> None:
        """Internal: clear faults and reseat commanded position to actual position.

        The reseat move prevents the arm from snapping to a stale commanded
        target after fault clearance — it commands the arm to its *current*
        actual position at a very slow speed.
        """
        self._arm.clean_error()
        self._arm.clean_warn()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(_RECOVERY_SETTLE_S)
        code, angles = self._arm.get_servo_angle(is_radian=True)
        if code == 0 and angles:
            self._arm.set_servo_angle(
                angle=angles[:6],
                speed=_RECOVERY_SPEED,
                mvacc=_RECOVERY_ACC,
                wait=True,
                is_radian=True,
            )
        log.info("Arm recovery complete")

    def recover(self) -> None:
        """Public recovery entry point — delegates to ``_recover``."""
        self._recover()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def destroy(self) -> None:
        self._arm.disconnect()
