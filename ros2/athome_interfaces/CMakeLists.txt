cmake_minimum_required(VERSION 3.8)
project(athome_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObservedObject.msg"
  "msg/ObservedObjectArray.msg"
  "msg/TargetResult.msg"
  "action/NavigateToGoal.action"
  "action/SearchObjects.action"
  "srv/ParseCommand.srv"
  DEPENDENCIES geometry_msgs nav_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
