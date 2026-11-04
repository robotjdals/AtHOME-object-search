"""Does the real-time detector see this object? No robot, SLAM or map:

    set -a && . ./.env && set +a        # ATHOME_GPU_SSH, ATHOME_CLIP_PYTHON
    ros2 launch athome_bringup perception_check.launch.py
    # http://localhost:8090
    ros2 launch athome_bringup perception_check.launch.py detector:=cgd     # perception's CG-D instead of YOLOE
    # CG-D: installed on the GPU server under /data/athome/cgd_test, slower (about 5 s per image)

Starts the D435 colour node (perception's v4l2_rgbd_node.py, ~/athome; camera:=false
if it already runs on this domain), the detection server on the GPU server with
its tunnel, the CLIP text server and the page (athome_ros.perception_check). Runs on its own
ROS domain, so it does not touch a robot stack; only one process can read the
camera. Ctrl-C stops the detection server, so the shared GPU is returned.
"""

import os
import shlex
import subprocess
from pathlib import Path

import athome
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable, UnsetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessIO, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

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
CAMERA = Path.home() / "athome/semantic_geometry_ws/tools/robot_live/v4l2_rgbd_node.py"
CALIB = REPO / "data/real_run6/real_run6_406_20261103/robot_recording/calibration.yaml"


def _setup(context):
    ssh = os.environ.get("ATHOME_GPU_SSH")
    if not ssh:
        raise RuntimeError("ATHOME_GPU_SSH가 없음: set -a && . ./.env && set +a")
    ssh = shlex.split(ssh)
    cgd = LaunchConfiguration("detector").perform(context) == "cgd"
    start, stop = (START_CGD, STOP_CGD) if cgd else (START, STOP)
    clip = ExecuteProcess(
        cmd=[os.environ.get("ATHOME_CLIP_PYTHON", "python3"), "-u", str(REPO / "scripts/serve_clip_text.py"),
             "--host", "127.0.0.1", "--port", "8100"],
        name="clip_text_server", output="screen")
    page = Node(package="athome_ros", executable="perception_check", output="screen",
                parameters=[{"detector": "cgd" if cgd else "yoloe"}])

    def on_clip_output(event):
        if b"serving" in event.text:
            return [LogInfo(msg="CLIP 문자 서버 준비됨: 화면 시작"), page]

    def stop_server(event, ctx):
        subprocess.run(["ssh", "-o", "ConnectTimeout=10", *ssh, stop], timeout=20, check=False)
        return [LogInfo(msg="GPU 서버의 검출 서버 종료")]

    return [
        ExecuteProcess(cmd=["ssh", "-o", "ConnectTimeout=10", *ssh, start], name="detector_server_start", output="screen"),
        ExecuteProcess(cmd=["ssh", "-N", "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes",
                            "-L", "18200:127.0.0.1:8200", *ssh], name="gpu_tunnel", output="screen"),
        clip,
        ExecuteProcess(cmd=["python3", str(CAMERA), "--width", "1280", "--height", "720", "--fps", "10",
                            "--no-depth", "--calib", str(CALIB)],
                       name="camera", output="screen", condition=IfCondition(LaunchConfiguration("camera"))),
        RegisterEventHandler(OnProcessIO(target_action=clip, on_stdout=on_clip_output)),
        RegisterEventHandler(OnShutdown(on_shutdown=stop_server)),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("detector", default_value="yoloe", description="yoloe | cgd"),
        # Its own graph: nothing here reaches a robot stack on the team domain.
        UnsetEnvironmentVariable("RMW_IMPLEMENTATION"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", "78"),
        SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
        OpaqueFunction(function=_setup),
    ])
