"""Search with real-time perception, everything but the operator console:

    set -a && . ./.env && set +a        # API keys, ATHOME_GPU_SSH, ATHOME_CLIP_PYTHON
    ros2 launch athome_bringup realtime_search.launch.py robot_config:=$PWD/configs/robot/room412_v2.yaml \
        [rviz:=true] [foxglove:=true]
    # commands, stop, manual checks: http://localhost:8080 (console:=false: ros2 run athome_ros command_cli)

Starts, in order:
  1. the detection server on the GPU server (scripts/serve_detector.py via
     /data/athome/perception/run_detector_server.sh; kept if already running),
  2. one SSH tunnel: detection server (local 18200) and LLM server (local 16007),
  3. the CLIP text server of the search's CLIP check (scripts/serve_clip_text.py),
  4. the real-time perception node (ros2/athome_perception),
  5. search.launch.py once the text server serves (the CLIP check embeds
     its vocabulary at startup).
On shutdown (Ctrl-C) the detection server is stopped, so the shared GPU is
returned. foxglove:=true also starts foxglove_bridge (ws://localhost:8765)
for the Foxglove dashboard (ros2/athome_bringup/foxglove/athome_search.json).

Environment (not in the repository):
  ATHOME_GPU_SSH      ssh arguments of the GPU server, e.g. "-p <port> <user>@<host>"
  ATHOME_CLIP_PYTHON  python of an environment with open_clip (default python3)
"""

import os
import shlex
import subprocess
from pathlib import Path

import athome
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessIO, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

REPO = Path(athome.__file__).resolve().parents[2]
SERVER_DIR = "/data/athome/perception"
START = (f"cd {SERVER_DIR} && if ! kill -0 $(cat out/detector_server.pid 2>/dev/null) 2>/dev/null; then "
         "nohup ./run_detector_server.sh > out/detector_server.log 2>&1 < /dev/null & "
         "echo $! > out/detector_server.pid; fi")
STOP = f"kill $(cat {SERVER_DIR}/out/detector_server.pid) 2>/dev/null; true"


def _setup(context):
    ssh = os.environ.get("ATHOME_GPU_SSH")
    if not ssh:
        raise RuntimeError("ATHOME_GPU_SSH가 없음: .env에 GPU 서버의 ssh 인자를 넣고 set -a && . ./.env && set +a")
    ssh = shlex.split(ssh)
    share = get_package_share_directory("athome_bringup")
    perception = LaunchConfiguration("perception_config").perform(context)
    clip = ExecuteProcess(
        cmd=[os.environ.get("ATHOME_CLIP_PYTHON", "python3"), "-u", str(REPO / "scripts/serve_clip_text.py"),
             "--host", "127.0.0.1", "--port", "8100"],
        name="clip_text_server", output="screen")
    search = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "search.launch.py")),
        launch_arguments={"robot_config": LaunchConfiguration("robot_config"),
                          "rviz": LaunchConfiguration("rviz")}.items())

    def on_clip_output(event):
        if b"serving" in event.text:
            return [LogInfo(msg="CLIP 문자 서버 준비됨: 탐색 서버 시작"), search]

    def stop_server(event, ctx):
        subprocess.run(["ssh", "-o", "ConnectTimeout=10", *ssh, STOP], timeout=20, check=False)
        return [LogInfo(msg="GPU 서버의 검출 서버 종료")]

    dashboard = []
    if LaunchConfiguration("foxglove").perform(context) == "true":
        # Images only as JPEG (the raw camera image is ~2.7 MB per frame).
        dashboard = [Node(package="foxglove_bridge", executable="foxglove_bridge", output="screen",
                          parameters=[{"port": 8765, "address": "0.0.0.0",
                                       "topic_whitelist": ["^/(?!.*image_raw$).*"]}])]
    if LaunchConfiguration("console").perform(context) == "true":
        dashboard.append(Node(package="athome_ros", executable="operator_console", output="screen",
                              parameters=[{"robot_config": LaunchConfiguration("robot_config")}]))
    return dashboard + [
        ExecuteProcess(cmd=["ssh", "-o", "ConnectTimeout=10", *ssh, START], name="detector_server_start",
                       output="screen"),
        ExecuteProcess(cmd=["ssh", "-N", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
                            "-L", "18200:127.0.0.1:8200", "-L", "16007:127.0.0.1:6007", *ssh],
                       name="gpu_tunnel", output="screen"),
        clip,
        Node(package="athome_perception", executable="realtime_perception", output="screen",
             parameters=[{"config": perception}] if perception else []),
        RegisterEventHandler(OnProcessIO(target_action=clip, on_stdout=on_clip_output)),
        RegisterEventHandler(OnShutdown(on_shutdown=stop_server)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_config"),
        DeclareLaunchArgument("rviz", default_value="false"),
        # foxglove_bridge for the dashboard (apt: ros-humble-foxglove-bridge)
        DeclareLaunchArgument("foxglove", default_value="false"),
        # Operator console web page (athome_ros.operator_console)
        DeclareLaunchArgument("console", default_value="true"),
        # '' = configs/perception/realtime.yaml
        DeclareLaunchArgument("perception_config", default_value=""),
        OpaqueFunction(function=_setup),
    ])
