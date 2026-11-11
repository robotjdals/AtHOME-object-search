"""Search with real-time perception, everything but the operator console:

    set -a && . ./.env && set +a        # API keys, ATHOME_GPU_SSH, ATHOME_CLIP_PYTHON
    ros2 launch athome_bringup realtime_search.launch.py robot_config:=$PWD/configs/robot/room412_v2.yaml \
        [rviz:=true] [foxglove:=true]
    # commands, stop, manual checks: http://localhost:8080 (console:=false: ros2 run athome_ros command_cli;
    # console_host:=0.0.0.0: from other computers, with the access token in the logged URL)

Starts, in order:
  1. the detection server on the GPU server (scripts/serve_detector.py via
     /data/athome/perception/run_detector_server.sh; kept if already running),
     or with detector_host:=local CG-D on this computer's GPU
     (athome.perception.local_cgd),
  2. one SSH tunnel: detection server (local 18200, not with detector_host:=local)
     and LLM server (local 16007),
  3. the CLIP text server of the search's CLIP check (scripts/serve_clip_text.py,
     on the CPU with detector_host:=local: the GPU is full),
  4. the real-time perception node (ros2/athome_perception),
  5. search.launch.py once the text server serves (the CLIP check embeds
     its vocabulary at startup).
On shutdown (Ctrl-C) the detection server is stopped, so the shared GPU is
returned (detector_host:=local leaves the GPU server alone). foxglove:=true also starts foxglove_bridge (ws://localhost:8765)
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
from athome.perception import local_cgd
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
CGD_DIR = "/data/athome/cgd_test"
START_CGD = (f"cd {CGD_DIR}/run && if ! kill -0 $(cat server.pid 2>/dev/null) 2>/dev/null; then "
             f"(source {CGD_DIR}/activate.sh && cd {CGD_DIR}/run && export GSA_RAM_GPU=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && "
             f"exec python serve_cgd_http.py --robot-live {CGD_DIR}/code/semantic_geometry_ws/tools/robot_live --port 8200) "
             "> server.log 2>&1 < /dev/null & echo $! > server.pid; fi")
STOP_CGD = f"kill $(cat {CGD_DIR}/run/server.pid) 2>/dev/null; true"


def _setup(context):
    ssh = os.environ.get("ATHOME_GPU_SSH")
    if not ssh:
        raise RuntimeError("ATHOME_GPU_SSH가 없음: .env에 GPU 서버의 ssh 인자를 넣고 set -a && . ./.env && set +a")
    ssh = shlex.split(ssh)
    share = get_package_share_directory("athome_bringup")
    cgd = LaunchConfiguration("detector").perform(context) == "cgd"
    start, stop = (START_CGD, STOP_CGD) if cgd else (START, STOP)
    local = LaunchConfiguration("detector_host").perform(context) == "local"
    if local and not cgd:
        raise RuntimeError("detector_host:=local은 detector:=cgd 전용")
    perception = LaunchConfiguration("perception_config").perform(context) or (
        str(REPO / "configs/perception/realtime_cgd.yaml") if cgd else "")
    clip = ExecuteProcess(
        cmd=[os.environ.get("ATHOME_CLIP_PYTHON", "python3"), "-u", str(REPO / "scripts/serve_clip_text.py"),
             "--host", "127.0.0.1", "--port", "8100", *(["--device", "cpu"] if local else [])],
        name="clip_text_server", output="screen")
    search = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "search.launch.py")),
        launch_arguments={"robot_config": LaunchConfiguration("robot_config"),
                          "rviz": LaunchConfiguration("rviz"),
                          "planner_variant": LaunchConfiguration("planner_variant")}.items())

    def on_clip_output(event):
        if b"serving" in event.text:
            return [LogInfo(msg="CLIP 문자 서버 준비됨: 탐색 서버 시작"), search]

    def stop_server(event, ctx):
        subprocess.run(["ssh", "-o", "ConnectTimeout=10", *ssh, stop], timeout=20, check=False)
        return [LogInfo(msg="GPU 서버의 검출 서버 종료")]

    if local:
        command, env = local_cgd.server_command(REPO, os.environ.get("ATHOME_CLIP_PYTHON", "python3"), Path.home())
        detector = [ExecuteProcess(cmd=command, additional_env=env, name="local_cgd_server", output="screen")]
        forwards = ["-L", "16007:127.0.0.1:6007"]
        shutdown = []
    else:
        detector = [ExecuteProcess(cmd=["ssh", "-o", "ConnectTimeout=10", *ssh, start], name="detector_server_start",
                                   output="screen")]
        forwards = ["-L", "18200:127.0.0.1:8200", "-L", "16007:127.0.0.1:6007"]
        shutdown = [RegisterEventHandler(OnShutdown(on_shutdown=stop_server))]

    dashboard = []
    if LaunchConfiguration("foxglove").perform(context) == "true":
        # Images only as JPEG (the raw camera image is ~2.7 MB per frame).
        dashboard = [Node(package="foxglove_bridge", executable="foxglove_bridge", output="screen",
                          parameters=[{"port": 8765, "address": "0.0.0.0",
                                       "topic_whitelist": ["^/(?!.*image_raw$).*"]}])]
    if LaunchConfiguration("console").perform(context) == "true":
        dashboard.append(Node(package="athome_ros", executable="operator_console", output="screen",
                              parameters=[{"robot_config": LaunchConfiguration("robot_config"),
                                           "host": LaunchConfiguration("console_host")}]))
    return dashboard + detector + [
        ExecuteProcess(cmd=["ssh", "-N", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
                            *forwards, *ssh],
                       name="gpu_tunnel", output="screen"),
        clip,
        Node(package="athome_perception", executable="realtime_perception", output="screen",
             parameters=[{"config": perception}] if perception else []),
        RegisterEventHandler(OnProcessIO(target_action=clip, on_stdout=on_clip_output)),
    ] + shutdown


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_config"),
        DeclareLaunchArgument("rviz", default_value="false"),
        # foxglove_bridge for the dashboard (apt: ros-humble-foxglove-bridge)
        DeclareLaunchArgument("foxglove", default_value="false"),
        # Operator console web page (athome_ros.operator_console)
        DeclareLaunchArgument("console", default_value="true"),
        # 0.0.0.0: other computers too, with the access token in the logged URL
        DeclareLaunchArgument("console_host", default_value="127.0.0.1"),
        # cgd: perception's CG-D without RAM (configs/perception/realtime_cgd.yaml, models load ~3 min)
        DeclareLaunchArgument("detector", default_value="cgd", description="cgd | yoloe"),
        # local: CG-D on this computer's GPU (1.48 s per image; lab 1.41 s when free, 3.11 s when busy)
        DeclareLaunchArgument("detector_host", default_value="server", description="server | local"),
        # '' = the detector's config (yoloe: configs/perception/realtime.yaml)
        DeclareLaunchArgument("perception_config", default_value=""),
        # trained: the planner adapters; base: the untrained base model for every stage (evaluation)
        DeclareLaunchArgument("planner_variant", default_value="trained", description="trained | base"),
        OpaqueFunction(function=_setup),
    ])
