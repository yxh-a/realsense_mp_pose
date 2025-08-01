cmake_minimum_required(VERSION 3.8)
project(ik_solver_moveit)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(srdfdom REQUIRED)
find_package(urdf REQUIRED)
find_package(yaml-cpp REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
add_executable(ik_solver src/ik_solver.cpp)

ament_target_dependencies(ik_solver
  rclcpp
  moveit_core
  geometry_msgs
  tf2_geometry_msgs
  ament_index_cpp
  srdfdom
  urdf
  yaml-cpp
  moveit_ros_planning
)

target_link_libraries(
  ik_solver
  ${catkin_LIBRARIES}
  yaml-cpp
)

# Install the executable
install(TARGETS ik_solver
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
