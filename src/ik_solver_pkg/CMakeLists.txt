cmake_minimum_required(VERSION 3.5)
project(ik_solver_pkg)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED) ##
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(urdf REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# For service generation
find_package(rosidl_default_generators REQUIRED)

include_directories(include)

add_executable(urdf_test_node src/ik_solver_node.cpp)

ament_target_dependencies(urdf_test_node
  rclcpp
  kdl_parser
  orocos_kdl
  urdf
  geometry_msgs
  trajectory_msgs
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  #ik_solver_pkg__rosidl_typesupport_cpp
)



# Add service files
set(SRV_FILES
  "srv/SetTargetPose.srv"
)

# ─── Service Interface Generation ──────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetTargetPose.srv"
  DEPENDENCIES sensor_msgs geometry_msgs builtin_interfaces
  #ADD_TYPESUPPORT c
)

# ✅ Link generated C++ typesupport
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

target_link_libraries(urdf_test_node ${cpp_typesupport_target})


# ✅ Add this to generate Python modules
# ament_python_install_package(${PROJECT_NAME})

install(TARGETS
  urdf_test_node
  DESTINATION lib/${PROJECT_NAME}
)

# ✅ This must come AFTER rosidl_generate_interfaces()
ament_export_dependencies(rosidl_default_runtime)

ament_package()
