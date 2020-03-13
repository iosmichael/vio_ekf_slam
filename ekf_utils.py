import numpy as np

"""
All transformation functions:
- Stereo Camera Functions
"""
# ========== STEREO-CAMERA METHODS ==========
def get_stereo_M(K, b):
    """
    Initialization of the stereo camera M matrix:
    :param: K - camera calibration matrix, b - stereo baseline (in meters)
    :return: M - stereo matrix for 3D - 2D projection
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, -1], K[1, -1]
    M = np.array([[fx, 0, cx, 0],
                  [0, fy, cy, 0],
                  [fx, 0, cx, -fx * b],
                  [0, fy, cy, 0]])
    return M

def get_oRr():
    """
    get the rotation matrix from the optical frame to the regular frame
    :return: the transpose of the rotation matrix R from regular to optical
    """
    R = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]])
    return R.T

def get_oTi():
    R = get_oRr()
    T = np.eye(4)
    T[:3, :3] = R
    return T

def get_P():
    P = np.zeros((3,4))
    P[:3, :3] = np.eye(3)
    return P

def stereo_pts_init(M, pts, cam_T_imu):
    """
    Initialization of the 3D point estimation
    - [uL, vL, uR, vR] = M^-1 * z * [x, y, z, 1]

    :param: a shape of 4 x N with [:2, :] = [uL, vL] and [2:, :] = [uR, vR]
    :return: 4 x N matrix for homogeneous coordinates of 3D pts
    """
    # calculate the disparity between left to the right: uL - uR = 1/z * fx b
    disparity, fx_b = pts[0, :] - pts[2, :], -M[2, -1]
    # calculate the depth estimation of the 3D points M(2,3)/uL-uR
    z = fx_b / disparity
    # since M is uninvertible, we need to solve the equation manually
    pts_3D = np.ones((4, pts.shape[1]))
    pts_3D[2, :] = z
    # x = uL * z / fx, y = uR * z / fy
    pts_3D[0, :] = (pts[0, :] * z - M[0,2] * z) / M[0,0]
    pts_3D[1, :] = (pts[1, :] * z - M[1,2] * z) / M[1,1]
    # calculate the [x, y, z, 1] in the optical frame
    X_r = np.linalg.inv(cam_T_imu) @ pts_3D
    return X_r

def homogenize(pts):
    return np.vstack((pts, np.ones((1, pts.shape[1]))))

def dehomogenize(pts):
    return pts[:3, :] / pts[-1, :]

def J_homo(q):
    """
    Calculating d_pi/d_q, the jacobian of the homogeneous
    :param: a shape of 4
    :return: a shape of 4 x 4 x N
    """
    assert q.shape[0] == 4 and len(q.shape) == 2
    J_q = np.eye(4)
    J_q[2, 2] = 0
    J_q[0, 2] = -q[0] / q[2]
    J_q[1, 2] = -q[1] / q[2]
    J_q[-1, 2] = -q[3] / q[2]
    return J_q / q[2]

def cov_homo(s):
    """
    Calculate the covariance of homogeneous coordinates
    :param: s: 4 x 1
    :return: 4 x 4
    """
    # make sure the scale is 1
    s = s / s[-1, :]
    cov = np.zeros((4, 6))
    cov[:3, :3] = np.eye(3)
    cov[:3, 3:] = -SO3_skew(s)
    return cov

# ======= IMU METHODS =======

def imu_T_cam(T):
    """
    get the transformation from IMU world frame to camera regular frame
    :return: the inverse of the IMU transformation matrix
    """
    return np.linalg.inv(T)

# ========== SO3 METHODS ==========

def SO3_exp(theta):
    """
    get the exponential map following the Rodrigues Formulas
    :param: angle-axis representation of the rotation matrix
    :return: the exponential mapping of theta (angle * axis) 3x3 rotation matrix
    """
    skew_theta, norm_theta = SO3_skew(theta), np.linalg.norm(theta)
    R = np.eye(3) + \
         (np.sin(norm_theta)/norm_theta) * skew_theta + \
         ((1-np.cos(norm_theta))/(norm_theta ** 2)) * (skew_theta @ skew_theta)
    assert(np.allclose(R.T @ R, np.eye(3)))
    assert(np.allclose(np.linalg.det(R), 1))
    return R

def SO3_angle_axis(R):
    """
    get the angle-axis map of the rotation matrix
    :param: rotation matrix
    :return: angle-axis representation of the rotation matrix
    """
    angle = np.arccos((np.trace(R) - 1)/2)
    axis = 1/(2 * np.sin(angle)) * np.array([R[2,1] - R[1,2],
                                             R[0,2] - R[2,0], 
                                             R[1,0] - R[0,1]])
    theta = axis * angle
    skew_theta = angle / (2 * np.sin(angle)) * (R - R.T)
    # testing on skew function
    assert np.allclose(SO3_skew(theta), skew_theta)
    assert np.allclose(np.linalg.norm(theta), angle)
    return axis * angle

def SO3_yawpitch2R(yaw, pitch):
    """
    Yaw and Pitch angles to construct the rotation matrix
    :param: yaw angle around z-axis, pitch angle around y-axis
    :return: SO(3) rotation matrix
    """
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                        [np.sin(yaw), np.cos(yaw), 0], 
                        [0, 0, 1]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0], 
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R = R_yaw @ R_pitch
    return R

def SO3_J_L(theta):
    """
    The left Jacobian
    :param: theta 3x1
    """
    skew_theta, norm_theta = SO3_skew(theta), np.linalg.norm(theta)
    return np.eye(3) \
        + (1- np.cos(norm_theta))/(norm_theta ** 2) * skew_theta \
        + (norm_theta - np.sin(norm_theta))/(norm_theta ** 3) * (skew_theta @ skew_theta)
    
# for testing purposes
def SO3_inv_J_L(theta):
    skew_theta, norm_theta = SO3_skew(theta), np.linalg.norm(theta)
    return np.eye(3) \
        - 1/2 * skew_theta + ((1+np.cos(norm_theta))/(norm_theta ** 2) \
        -(1)/(2 * norm_theta * np.sin(norm_theta))) * (skew_theta @ skew_theta)

def SO3_skew(w):
    """
    :param: a shape of 3 x 1
    :return: 3 x 3 skew symmetric matrix
    """
    x, y, z = w[0], w[1], w[2]
    A = np.array([[0, -z, y],[z, 0, -x], [-y, x, 0]])
    return A

# ========== SE3 METHODS ==========

def SE3_skew(v):
    """
    hat map of special euclidean group
    :param: [rho, theta], 6x1 
    :return: transformation T, 4x4
    """
    rho, theta = v[:3], v[3:]
    T = np.zeros((4,4))
    T[:3, :3] = SO3_skew(theta)
    T[:3, -1] = rho
    return T

def SE3_exp(v):
    """
    :param: [rho, theta], 6x1 
    :return: transformation T, 4x4
    """
    rho, theta = v[:3], v[3:]
    T = np.eye(4)
    T[:3, :3] = SO3_skew(theta)
    T[:3, -1] = (SO3_J_L(theta) @ rho.reshape(3,1)).flatten()
    return T

def SE3_cov(v):
    """
    perturbation map of special euclidean group
    :param: [rho, theta], 6x1
    :return: covariance of SE3, 6x6
    """
    rho, theta = v[:3], v[3:]
    skew_rho, skew_theta = SO3_skew(rho), SO3_skew(theta)
    cov = np.zeros((6,6))
    cov[:3, :3] = cov[3:, 3:] = skew_theta
    cov[:3, 3:] = skew_rho
    return cov

# ========== SE(3) Perturbation METHODS ==========
def u_hat(linear, angular):
    """
    :param: v - linear velocity from the IMU data, w - angular velocity from the IMU data
    :return: a tuple of (u, u_hat, u_hat_pert) for prediction step
    """
    skew = np.zeros((4,4))
    skew[:3, :3] = SO3_skew(angular)
    skew[:3, -1] = linear
    return skew

def u_cur(linear, angular):
    cur = np.zeros((6,6))
    skew_linear, skew_angular = SO3_skew(linear), SO3_skew(angular)
    cur[:3, :3] = cur[3:, 3:] = skew_angular
    cur[:3, 3:] = skew_linear
    return cur

def get_homo_cur(s):
    """
    transformation used for calculating the EKF of the points estimation
    """
    s_pert = np.zeros((4, 6))
    s_pert[:3, :3], s_pert[:3, 3:] = np.eye(3), -SO3_skew(s)
    return s_pert

def divide_z(A):
    assert A.shape[0] == 4
    return A / A[2, :]