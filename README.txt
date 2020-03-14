README

====== FILE DESCRIPTION ============

main.py - the python execution file for running EKF SLAM, which takes dataset from dirname in the configuration dictionary

ekf_vio.py - contains the EKF class definition and update procedure

ekf_utils.py - contains the transformation and functions used in EKF estimation

utils.py - contains the load_data function and visualization function

test.py - contains the testing unit for SO3 rotation matrix exponential map

====== DATASET DESCRIPTION ==========

"data" folder contains the landmark camera features and IMU used in this project 