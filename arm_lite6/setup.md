# Lite 6 IK Setup

## Quick Start (2 terminals)

### Terminal 1 — launch everything
```
cd ~/lite6_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch my_lite6_control lite6_full.launch.py
```

This single command:
1. Starts MoveIt Servo + real robot connection
2. Waits 5 s, then sequentially runs robot init (clear error → enable → mode 1 → state 0)
3. Starts the IK solver node

Override the robot IP if needed:
```
ros2 launch my_lite6_control lite6_full.launch.py robot_ip:=192.168.1.200
```

### Terminal 2 — send target positions
```
cd ~/lite6_ws/src/my_lite6_control/my_lite6_control/
python3 target.py
```
Enter `x y z` in metres at the prompt, e.g. `0.3 0.0 0.3`.

---

## Error Recovery

If an error occurs, re-run the init sequence manually:
```
ros2 service call /ufactory/clean_error xarm_msgs/srv/Call "{}"
ros2 service call /ufactory/motion_enable xarm_msgs/srv/SetInt16ById "{id: 8, data: 1}"
ros2 service call /ufactory/set_mode xarm_msgs/srv/SetInt16 "{data: 1}"
ros2 service call /ufactory/set_state xarm_msgs/srv/SetInt16 "{data: 0}"
```

Or re-run the helper directly:
```
bash ~/lite6_ws/src/my_lite6_control/my_lite6_control/scripts/init_robot.sh
```

---

## Gripper Control
```
ros2 service call /ufactory/open_lite6_gripper  xarm_msgs/srv/Call "{}"
ros2 service call /ufactory/close_lite6_gripper xarm_msgs/srv/Call "{}"
ros2 service call /ufactory/stop_lite6_gripper  xarm_msgs/srv/Call "{}"
```

## Move to Init Pose
```
ros2 topic pub --once /lite6_traj_controller/joint_trajectory \
  trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['joint1','joint2','joint3','joint4','joint5','joint6'], \
    points: [{positions: [0.0, 0.0, 1.57, 0.0, 1.57, 0.0], \
              time_from_start: {sec: 3, nanosec: 0}}]}"
```

---

## Dependency Installation

### 1. ROS 2

Install ROS 2 Jazzy (Ubuntu 24.04). Adjust the distro name for your OS.

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools python3-rosdep
```

> **Broken package state:** After adding the ROS 2 apt repo, if existing packages have unmet
> dependencies, run the following in order before continuing:
> ```bash
> sudo apt install -y python3-catkin-pkg python3-catkin-pkg-modules
> sudo apt --fix-broken install
> sudo dpkg --configure -a
> ```

Add to `~/.bashrc` so every terminal sources ROS 2 automatically:
```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 2. MoveIt 2 + Servo + ros2_control

```bash
sudo apt install -y ros-jazzy-moveit ros-jazzy-moveit-servo \
                    ros-jazzy-ros2-control ros-jazzy-ros2-controllers
```

### 3. xarm_ros2

Clone UFactory's ROS 2 driver. Use the `jazzy` branch — the default `humble` branch
will fail to compile against Jazzy's ros2_control headers.

```bash
mkdir -p ~/xarm_ros2/src && cd ~/xarm_ros2/src
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive
cd xarm_ros2 && git checkout jazzy && git submodule update --recursive
cd ~/xarm_ros2
```

Initialize rosdep (first time only). If `rosdep update` fails with SSL errors, set
`SSL_CERT_FILE` to point at the system bundle:

```bash
sudo rosdep init
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt rosdep update
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    rosdep install --from-paths src --ignore-src -r -y
```

Build, skipping the Gazebo and RealSense simulation packages (not needed for real robot):

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install \
    --packages-skip realsense_gazebo_plugin xarm_gazebo d435i_xarm_setup mbot_demo
echo "source ~/xarm_ros2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4. Python dependencies

`numpy` is already installed as a system package with ROS 2 Jazzy. No venv needed.

If it is somehow missing:
```bash
sudo apt install -y python3-numpy
```

### 5. my_lite6_control package

The workspace lives at `~/lite6_ws`. Copy the files from this repo into it:

```bash
mkdir -p ~/lite6_ws/src/my_lite6_control/my_lite6_control
cp -r arm_lite6/* ~/lite6_ws/src/my_lite6_control/my_lite6_control/
```

The package also needs `package.xml`, `setup.py`, a resource marker, an `__init__.py`,
and executable shim scripts so `ros2 run` can find the nodes. These are in
`arm_lite6/` alongside the Python files and are copied above.

Build:
```bash
cd ~/lite6_ws
source ~/xarm_ros2/install/setup.bash
colcon build --symlink-install
echo "source ~/lite6_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
ros2 pkg executables my_lite6_control
# should print:
# my_lite6_control custom_ik
# my_lite6_control target
```

> `--symlink-install` means edits to the Python source files take effect without rebuilding.
