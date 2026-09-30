"""Record a search run for the run report.

    ros2 launch athome_bringup record.launch.py [out:=outputs/runs/<name>] [perception:=true]

Start it before the first command, stop it (Ctrl-C) after the result, then
    ros2 run athome_ros run_report <bag> --robot-config $PWD/configs/robot/demo.yaml

Recorded: TF (robot path), search events and planner queries (the report),
action status and feedback, /map, search markers and logs (replay in RViz).
Perception messages carry CLIP features (large): only with perception:=true.
"""

import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

TOPICS = [
    "/tf", "/tf_static",
    "/athome/search/events", "/athome/planner/decision",
    "/athome/search_objects/_action/feedback", "/athome/search_objects/_action/status",
    "/athome/navigate_to_goal/_action/feedback", "/athome/navigate_to_goal/_action/status",
    "/map", "/athome/viz/search", "/rosout",
]
PERCEPTION = ["/athome/perception/objects"]


def launch_setup(context):
    out = LaunchConfiguration("out").perform(context) or (
        "outputs/runs/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    topics = list(TOPICS)
    if LaunchConfiguration("perception").perform(context) == "true":
        topics += PERCEPTION
    topics += LaunchConfiguration("extra_topics").perform(context).split()
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "-o", out, *topics], output="screen")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("out", default_value="", description="bag directory (new)"),
        DeclareLaunchArgument("perception", default_value="false"),
        DeclareLaunchArgument("extra_topics", default_value="",
                              description="more topics, space separated (e.g. /cmd_vel)"),
        OpaqueFunction(function=launch_setup),
    ])
