set(COMMON_HEADERS
        include/cv-utils.h
        include/app-utils.h
        include/log-utils.h
        include/string-utils.h)

set(COMMON_SRC
        cpp/cv-utils.cpp
#        cpp/app-utils.cpp
        cpp/log-utils.cpp
        cpp/string-utils.cpp)


set(FOCUS_STACKER_HEADERS
        include/glob/glob.hpp)

set(FOCUS_STACKER_SRC
        cpp/main.cpp
)


add_executable(focus-stacker ${FOCUS_STACKER_SRC} ${FOCUS_STACKER_HEADERS} ${COMMON_SRC} ${COMMON_HEADERS})
target_include_directories(focus-stacker PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(UNIX)
    target_link_libraries(focus-stacker PRIVATE ${OpenCV_LIBS} fmt::fmt ${Sfml} pthread)# rt)
else(WIN32)
    target_link_libraries(focus-stacker PRIVATE ${OpenCV_LIBS} fmt::fmt)
    # target_compile_options(boost-ipc-client PUBLIC /MTd)
endif()

