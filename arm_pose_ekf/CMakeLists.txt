cmake_minimum_required(VERSION 3.8)
project(arm_pose_ekf)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(arm_pose_ekf_node
  src/arm_pose_ekf_node.cpp
)

target_include_directories(arm_pose_ekf_node PRIVATE
  include
)

ament_target_dependencies(arm_pose_ekf_node
  ament_index_cpp
  geometry_msgs
  pinocchio
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  visualization_msgs
)

target_compile_features(arm_pose_ekf_node PUBLIC cxx_std_17)

install(TARGETS arm_pose_ekf_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
