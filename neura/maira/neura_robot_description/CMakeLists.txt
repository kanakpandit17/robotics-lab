cmake_minimum_required(VERSION 3.8)
project(neura_robot_description)

find_package(ament_cmake REQUIRED)

# Install everything so find-pkg-share can locate it
install(DIRECTORY maira7M
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
