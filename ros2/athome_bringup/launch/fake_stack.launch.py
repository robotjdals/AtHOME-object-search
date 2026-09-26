"""Fake motion + perception modules for testing the search executor without the robot."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="athome_ros", executable="fake_motion_server", output="screen"),
        Node(package="athome_ros", executable="fake_perception", output="screen"),
    ])
