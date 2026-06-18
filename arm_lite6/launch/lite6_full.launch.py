import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
_LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))

# Delay (seconds) before running robot init — give moveit_servo time to come up.
INIT_DELAY = 5.0


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.150',
        description='IP address of the Lite 6 controller',
    )

    robot_ip = LaunchConfiguration('robot_ip')

    # Use our local patched launch file instead of the xarm_moveit_servo chain.
    # The upstream chain (lite6_moveit_servo_realmove → _robot_moveit_servo_realmove)
    # loses the ros2_control_plugin argument due to Jazzy's scoped argument handling,
    # causing UFRobotFakeSystemHardware to be loaded regardless of what is passed.
    moveit_servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_LAUNCH_DIR, '_real_hardware_moveit_servo.launch.py')
        ),
        launch_arguments={
            'robot_ip':    robot_ip,
            'dof':         '6',
            'robot_type':  'lite',
            'hw_ns':       'ufactory',
            'limited':     'true',
            'add_gripper': 'true',
            'report_type': 'dev',
            'ros2_control_plugin': 'uf_robot_hardware/UFRobotSystemHardware',
        }.items(),
    )

    init_robot = TimerAction(
        period=INIT_DELAY,
        actions=[
            ExecuteProcess(
                cmd=['bash', os.path.join(SCRIPTS_DIR, 'init_robot.sh')],
                output='screen',
            )
        ],
    )

    ik_node = Node(
        package='my_lite6_control',
        executable='custom_ik',
        output='screen',
    )

    return LaunchDescription([
        robot_ip_arg,
        moveit_servo,
        init_robot,
        ik_node,
    ])
