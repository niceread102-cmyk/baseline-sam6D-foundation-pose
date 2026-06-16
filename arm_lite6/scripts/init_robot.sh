#!/usr/bin/env bash
# Sequential robot initialization — must run after moveit_servo is up.
set -e

echo "[init_robot] Clearing errors..."
ros2 service call /ufactory/clean_error xarm_msgs/srv/Call "{}"

echo "[init_robot] Enabling motion..."
ros2 service call /ufactory/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"

echo "[init_robot] Setting mode 1 (servo)..."
ros2 service call /ufactory/set_mode xarm_msgs/srv/SetInt16 "{data: 1}"

echo "[init_robot] Setting state 0 (ready)..."
ros2 service call /ufactory/set_state xarm_msgs/srv/SetInt16 "{data: 0}"

echo "[init_robot] Done."
