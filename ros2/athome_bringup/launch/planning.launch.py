"""Nav2 map server + planner server (global costmap) from the robot config.

    ros2 launch athome_bringup planning.launch.py robot_config:=$PWD/configs/robot/demo.yaml

Only these Nav2 parts are used (no bt_navigator / controller_server):
  /map                        nav2_map_server (skip with map_server:=false
                              if localization already serves the map)
  /global_costmap/costmap     static + inflation layers, footprint =
                              robot.footprint_m (or robot_radius =
                              robot.inflation_radius). MPPI should use this
                              same costmap for static obstacles.
  compute_path_to_pose        NavFn with A* (use_astar), used by search_server
"""

import tempfile

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from athome.config import load_robot_config


def nav2_params(cfg) -> dict:
    costmap = {
        "update_frequency": 1.0,
        "publish_frequency": 1.0,
        "global_frame": cfg.map_frame,
        "robot_base_frame": cfg.base_frame,
        "track_unknown_space": cfg.unknown_as_occupied,
        "always_send_full_costmap": True,
        "plugins": ["static_layer", "inflation_layer"],
        "static_layer": {
            "plugin": "nav2_costmap_2d::StaticLayer",
            "map_subscribe_transient_local": True,
        },
        "inflation_layer": {
            "plugin": "nav2_costmap_2d::InflationLayer",
            "inflation_radius": cfg.costmap_inflation_radius,
            "cost_scaling_factor": 3.0,
        },
    }
    # Rectangular robots give Nav2 the footprint polygon (it derives the
    # inscribed/circumscribed radii); circular ones the radius.
    if cfg.geometry is not None:
        costmap["footprint"] = str(cfg.geometry.polygon())
    else:
        costmap["robot_radius"] = cfg.inflation_radius
    return {
        "map_server": {"ros__parameters": {"yaml_filename": str(cfg.map_yaml)}},
        "planner_server": {"ros__parameters": {
            "expected_planner_frequency": 1.0,
            "planner_plugins": ["GridBased"],
            "GridBased": {
                "plugin": "nav2_navfn_planner/NavfnPlanner",
                "use_astar": True,
                "allow_unknown": not cfg.unknown_as_occupied,
                "tolerance": 0.0,
            },
        }},
        "global_costmap": {"global_costmap": {"ros__parameters": costmap}},
    }


def launch_setup(context):
    cfg = load_robot_config(LaunchConfiguration("robot_config").perform(context))
    params = tempfile.NamedTemporaryFile(
        "w", prefix="athome_nav2_", suffix=".yaml", delete=False)
    yaml.safe_dump(nav2_params(cfg), params)
    params.close()

    use_map_server = LaunchConfiguration("map_server").perform(context) == "true"
    managed = (["map_server"] if use_map_server else []) + ["planner_server"]
    nodes = [
        Node(package="nav2_planner", executable="planner_server", output="screen",
             parameters=[params.name]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_planning", output="screen",
             parameters=[{"autostart": True, "node_names": managed}]),
    ]
    if use_map_server:
        nodes.insert(0, Node(package="nav2_map_server", executable="map_server",
                             output="screen", parameters=[params.name]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_config"),
        DeclareLaunchArgument("map_server", default_value="true"),
        OpaqueFunction(function=launch_setup),
    ])
