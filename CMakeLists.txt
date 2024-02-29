cmake_minimum_required(VERSION 3.7)
project(cplus_test)

set(CMAKE_CXX_STANDARD 11)

# 指定文件夹位置
set(OPENCV_DIR /home/ayi/Downloads/opencv-4.5.4)
#set(Torch_DIR /home/ayi/libtorch/share/cmake/Torch)

# 自动查找包
#find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Eigen3 3.4.0 REQUIRED NO_MODULE)

# 添加源程序
add_executable(cplus_test main.cpp src/main_yolo.cpp)

# 添加头文件
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS} ./include)

# 加入库文件位置
target_link_libraries(cplus_test 
                      ${OpenCV_LIBS} 
                      ${TORCH_LIBRARIES})
target_link_libraries (cplus_test Eigen3::Eigen)