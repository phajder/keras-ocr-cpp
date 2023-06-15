cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(KerasOCR)

add_executable(KerasOCR keras_ocr.cpp)
#add_library(KerasOCR SHARED label_ocr.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(cppflow REQUIRED)

target_include_directories(
        KerasOCR PUBLIC
        cppflow::cppflow
)

target_link_libraries(
        KerasOCR PUBLIC
        cppflow::cppflow
        ${OpenCV_LIBRARIES}
)
# define the symbol stating we are using the declspec(dllexport) when
# building on windows
target_compile_features(KerasOCR PUBLIC cxx_std_17)
target_compile_definitions(KerasOCR PUBLIC "EXPORTING_LABEL_OCR")
target_compile_definitions(KerasOCR PUBLIC
                DETECTION_MODEL_PATH="${CMAKE_SOURCE_DIR}/../models/craft-detection"
                RECOGNITION_MODEL_PATH="${CMAKE_SOURCE_DIR}/../models/crnn-recognition"
        )

install(TARGETS KerasOCR)
