cmake_minimum_required(VERSION 3.8)
project(arm_controllers)


# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)

find_package(std_msgs REQUIRED)
find_package(arm_interfaces REQUIRED)

# Unitree SDK2 라이브러리
set(UNITREE_SDK_DIR "$ENV{HOME}/workspaces/unitree_sdk2")


# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


### arm_controller_node ###
add_executable(arm_controller_node
	src/arm_controller_node.cpp
)

target_include_directories(arm_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${UNITREE_SDK_DIR}/include
  ${UNITREE_SDK_DIR}/include/unitree/robot/channel
  ${UNITREE_SDK_DIR}/thirdparty/include
  ${UNITREE_SDK_DIR}/thirdparty/include/ddscxx
  $ENV{HOME}/unitree_ros2/cyclonedds_ws/install/include
)

link_directories(
  ${UNITREE_SDK_DIR}/lib/aarch64
  ${UNITREE_SDK_DIR}/thirdparty/ddsc/lib
  $ENV{HOME}/unitree_ros2/cyclonedds_ws/install/lib
)

target_link_libraries(arm_controller_node
  unitree_sdk2
  ddsc
  ddscxx
)


ament_target_dependencies(arm_controller_node
  rclcpp
  rclcpp_action

  std_msgs
  arm_interfaces
)


# install cpp executables
install(TARGETS
  arm_controller_node
	DESTINATION lib/${PROJECT_NAME})


# install launch param
# install (DIRECTORY launch param
#   DESTINATION share/${PROJECT_NAME}
# )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
