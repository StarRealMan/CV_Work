add_library(calib SHARED
            Zhang.cpp
            Confidence.cpp
            Optimizer.cpp)

target_link_libraries(calib
                      ${THIRD_PARTY_LIBS})
