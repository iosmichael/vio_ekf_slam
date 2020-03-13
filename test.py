import numpy as np 
from ekf_utils import *

# PASS
def test_exponential_logorithm_map_SO3():
    # initialize with random R
    R = SO3_yawpitch2R(np.pi/3, np.pi/3)
    theta = SO3_angle_axis(R)
    R_p = SO3_exp(theta)
    assert np.allclose(R, R_p)

# PASS
def test_left_jacobian_SO3():
    # initialize with random R
    R = SO3_yawpitch2R(np.pi/3, np.pi/3)
    theta = SO3_angle_axis(R)
    R_p = SO3_exp(theta)
    R_jl = np.eye(3) + SO3_skew(theta) @ SO3_J_L(theta)
    assert np.allclose(R_p, R_jl)

test_exponential_logorithm_map_SO3()
test_left_jacobian_SO3()