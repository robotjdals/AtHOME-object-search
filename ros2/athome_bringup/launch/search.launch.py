"""Search system on the robot (perception and MPPI are launched by their owners).

    ros2 launch athome_bringup search.launch.py robot_config:=$PWD/configs/robot/demo.yaml
    ros2 launch athome_bringup search.launch.py robot_config:=... rviz:=true

Before the first command, with localization, MPPI and perception up:
    ros2 run athome_ros preflight --ros-args -p robot_config:=$PWD/configs/robot/demo.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("athome_bringup")
    rviz_config = os.path.join(share, "rviz", "search.rviz")
    return LaunchDescription([
        DeclareLaunchArgument("robot_config"),
        DeclareLaunchArgument("rviz", default_value="false"),
        # false if localization already serves /map.
        DeclareLaunchArgument("map_server", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "planning.launch.py")),
            launch_arguments={
                "robot_config": LaunchConfiguration("robot_config"),
                "map_server": LaunchConfiguration("map_server"),
            }.items(),
        ),
        Node(
            package="athome_ros", executable="search_server", output="screen",
            parameters=[{"robot_config": LaunchConfiguration("robot_config")}],
            # Ctrl-C: the node cancels the motion and waits for the stop.
            sigterm_timeout="10", sigkill_timeout="10",
        ),
        Node(
            package="rviz2", executable="rviz2", arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
