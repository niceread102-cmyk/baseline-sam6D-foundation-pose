#!/usr/bin/env python3
"""
test_arm.py — Minimal arm connectivity and movement test.

Connects directly to the Lite 6 via the xArm Python SDK (no ROS required).

What it does:
  1. Connects to the robot and runs the init sequence
  2. Prints a brief diagnostic (firmware version, error state)
  3. Moves joints to the known home position
  4. Nudges joint 1 by TEST_DELTA radians (~3°) so you can watch it move
  5. Returns to home
  6. Cycles the gripper open → close → open

Safe to run with the arm at rest in a clear workspace.
"""
import logging
import sys

# ── Settings ─────────────────────────────────────────────────────────────────

ROBOT_IP    = "192.168.1.150"
HOME_ANGLES = [0.0, 0.7, 1.57, 0.0, 0.87, 0.0]   # radians
TEST_JOINT  = 0       # joint index to nudge (0 = base rotation, 1-based in xArm docs)
TEST_DELTA  = 0.05    # radians (~3°)
MOVE_SPEED  = 0.3     # rad/s — deliberately slow
MOVE_ACC    = 5.0     # rad/s²

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check(label: str, code) -> bool:
    """Log the return code and return True if successful (code == 0)."""
    if code not in (0, None):
        log.error("FAILED  %-30s  code=%s", label, code)
        return False
    log.info("OK      %s", label)
    return True


def move(arm, angles: list[float], label: str) -> bool:
    log.info("Moving → %s  %s", label, [round(a, 3) for a in angles])
    code = arm.set_servo_angle(
        angle=angles,
        speed=MOVE_SPEED,
        mvacc=MOVE_ACC,
        wait=True,
        is_radian=True,
    )
    return check(f"move {label}", code)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from xarm import XArmAPI

    log.info("Connecting to xArm at %s ...", ROBOT_IP)
    arm = XArmAPI(port=ROBOT_IP, is_radian=True)
    arm.connect()
    if not arm.connected:
        log.error("Connection failed — check IP and that the robot is powered on.")
        sys.exit(1)
    log.info("Connected.")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    log.info("=== Diagnostics ===")
    log.info("  firmware_version : %s", arm.version)
    log.info("  error_code       : %s", arm.error_code)
    log.info("  warn_code        : %s", arm.warn_code)
    log.info("  state            : %s", arm.state)

    # ── Init sequence ─────────────────────────────────────────────────────────
    log.info("=== Init ===")
    arm.clean_error()
    arm.clean_warn()
    check("motion_enable", arm.motion_enable(True))
    check("set_mode(0)",   arm.set_mode(0))
    check("set_state(0)",  arm.set_state(0))

    # ── Read current angles ───────────────────────────────────────────────────
    code, current = arm.get_servo_angle(is_radian=True)
    if code == 0:
        log.info("Current angles (rad): %s", [round(a, 3) for a in current[:6]])
    else:
        log.warning("get_servo_angle returned code=%d", code)

    # ── Move to home ──────────────────────────────────────────────────────────
    log.info("=== Moving to home ===")
    if not move(arm, HOME_ANGLES, "home"):
        log.error("Home move failed — aborting.")
        arm.disconnect()
        sys.exit(1)

    # ── Test nudge ────────────────────────────────────────────────────────────
    nudged = HOME_ANGLES.copy()
    nudged[TEST_JOINT] += TEST_DELTA
    log.info(
        "=== Nudging joint %d by %.3f rad (%.1f°) — watch the arm! ===",
        TEST_JOINT + 1, TEST_DELTA, TEST_DELTA * 57.3,
    )
    if not move(arm, nudged, "nudged"):
        log.error("Nudge failed.")

    # ── Return to home ────────────────────────────────────────────────────────
    log.info("=== Returning to home ===")
    move(arm, HOME_ANGLES, "home")

    # ── Gripper cycle ─────────────────────────────────────────────────────────
    log.info("=== Gripper cycle (open → close → open) ===")
    check("open_lite6_gripper",  arm.open_lite6_gripper())
    check("close_lite6_gripper", arm.close_lite6_gripper())
    check("open_lite6_gripper",  arm.open_lite6_gripper())

    arm.disconnect()
    log.info("Done. If the arm moved on the nudge and the gripper cycled, all is well.")


if __name__ == "__main__":
    main()
