add_executable(Calibration
               Calibration.cpp)

target_link_libraries(Calibration
                      calib
                      ${THIRD_PARTY_LIBS})
