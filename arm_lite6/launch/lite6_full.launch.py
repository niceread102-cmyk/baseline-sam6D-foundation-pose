import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')

# Delay (seconds) before running robot init — give moveit_servo time to come up.
INIT_DELAY = 5.0


def generate_launch_description():
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.1.150',
        description='IP address of the Lite 6 controller',
    )

    robot_ip = LaunchConfiguration('robot_ip')

    xarm_servo_share = get_package_share_directory('xarm_moveit_servo')
    moveit_servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(xarm_servo_share, 'launch',
                         'lite6_moveit_servo_realmove.launch.py')
        ),
        launch_arguments={
            'robot_ip':    robot_ip,
            'add_gripper': 'true',
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
