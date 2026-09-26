"""Fake motion + perception + Nav2 planning in the toy environment (see athome.testing.toy_env).

    ros2 launch athome_bringup fake_toy.launch.py robot_config:=$PWD/configs/robot/toy.yaml \
        moved:="['cup_2=7.0,2.4,0.5']"
"""

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    moved = LaunchConfiguration("moved")
    config = LaunchConfiguration("robot_config")
    return LaunchDescription([
        DeclareLaunchArgument("robot_config"),
        DeclareLaunchArgument("moved", default_value="['']"),
        Node(
            package="athome_ros", executable="fake_motion_server", output="screen",
            parameters=[{"initial_pose": [2.5, 3.0, 0.0], "robot_config": config}],
        ),
        # Nav2 map server + planner (fake motion provides the TF it needs).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("athome_bringup"), "launch", "planning.launch.py")),
            launch_arguments={"robot_config": config}.items(),
        ),
        Node(
            package="athome_ros", executable="fake_perception", output="screen",
            parameters=[{
                "world": "toy",
                "moved": ParameterValue(moved, value_type=List[str]),
            }],
        ),
    ])
