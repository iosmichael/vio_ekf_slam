import numpy as np
import scipy
from scipy import linalg
from ekf_utils import *
from matplotlib import pyplot as plt

pose_var = 1 ** 2
landmark_var = 1 ** 2

class EKF_VIO(object):
	"""
	Extended Kalman Filter for Visual Inertial Odemetry
	- K, stereo camera calibration matrix, b, stereo camera baseline (in meters)
	"""
	def __init__(self, config, K, b, cam_T_imu, num_landmark):
		self.ep = 0

		self.config = config
		self.M = get_stereo_M(K, b)
		self.cam_T_imu = cam_T_imu
		# 3D robot pose estimation (x, y, z, axis_x, axis_y, angle), where rotation uses angle axis representation
		# inverse pose: U in SE(3): 4x4
		self.pose = np.linalg.inv(np.eye(4))
		# 6 degrees of freedom covariance matrix: 6x6
		self.cov_pose2pose = np.eye(6) * pose_var
		
		# landmark position estimation
		self.landmark = np.ones((3, num_landmark)) * -1
		# 3M x 3M covariance matrix between landmarks
		self.cov_landmark2landmark = np.eye(num_landmark * 3) * landmark_var

		# history poses
		self.observed = np.zeros(num_landmark)
		self.poses = []
		self.robustness = []

	def get_pose_in_world(self):
		return np.linalg.inv(self.pose)

	def advance_one_step(self, tau, imu, features):
		"""
		:param: linear, angular (3 x 1), features (N x 4)
		"""
		self.ep += 1
		self.pose_predict(imu, tau)
		# keypoint observation: 1, 2, 3 ...
		ind = np.where(np.min(features, axis=0) != -1)
		# previously unobserved landmark: 2, 3, ...
		ind_unobs = np.where(self.observed == 0)
		ind_obs = np.where(self.observed == 1)
		unobs = np.intersect1d(ind, ind_unobs)
		obs = np.intersect1d(ind, ind_obs)
		# using the current updated pose to initialize previously unobserved points
		self.landmark[:, unobs] = self.landmark_init(features[:, unobs])
		if len(obs) > 0:
			self.joint_update(features[:, obs], obs, np.union1d(obs, np.setdiff1d(ind_obs,obs)[-100:]))
		# record the observed landmarks
		self.observed[unobs] = 1

	def pose_predict(self, imu, tau):
		"""
		:param: predict robot position based on IMU data, imu: (linear, angular)
		"""
		linear, angular = imu
		self.record_pose(self.pose)
		self.pose = linalg.expm(-tau * u_hat(linear, angular)) @ self.pose
		# 6x6 covariance matrix
		exp_cov = linalg.expm(-tau * u_cur(linear,angular))
		self.cov_pose2pose = exp_cov @ self.cov_pose2pose @ exp_cov.T + np.eye(6) * pose_var

	def joint_update(self, obs, ind_kpt, ind):
		"""
		:param: ind, the index of landmarks being observed, obs, the feature observation
		:return: updated pose and landmarks
		"""
		N, M = obs.shape[1], len(ind)
		# get the landmark statistics
		mu_landmark = self.landmark[:, ind].T.reshape(-1, 1)
		cov_mask = np.concatenate([ind * 3, ind * 3 + 1, ind * 3 + 2])
		cov_mask_x, cov_mask_y = np.meshgrid(cov_mask, cov_mask, sparse=False, indexing='xy')
		cov_landmark = self.cov_landmark2landmark[cov_mask_x, cov_mask_y]		
		
		X = self.pose @ homogenize(self.landmark[:, ind_kpt])
		z = self.M @ divide_z(self.cam_T_imu @ X)
		P = get_P()
		# 4 N x 3 M + 6
		H = np.zeros((4 * N, 3 * M + 6))
		# build diagonal Jacobian
		y = self.cam_T_imu @ self.pose @ P.T
		for i in range(N):
			# Hi: 4x3
			j = np.where(ind == ind_kpt[i])[0][0]
			assert ind_kpt[i] == ind[j]
			H[4*i:4*(i+1), 3*j:3*(j+1)] = self.M @ J_homo(self.cam_T_imu @ X[:, i].reshape(-1,1)) @ y
			H[4*i:4*(i+1), -6:] = self.M @ J_homo(self.cam_T_imu @ X[:, i].reshape(-1,1)) @ self.cam_T_imu @ get_homo_cur(X[:, i])
		
		joint_cov = np.zeros((3 * M + 6, 3 * M + 6))
		joint_cov[-6:, -6:], joint_cov[:-6, :-6] = self.cov_pose2pose, cov_landmark
		# 3 M + 6 x 4 N
		K = joint_cov @ H.T @ np.linalg.inv((H @ joint_cov @ H.T + np.eye(4 * N) * landmark_var))
		# use Kalman Gain to update the landmark
		mu_landmark = mu_landmark + K[:-6, :] @ (obs.T.reshape(-1, 1) - z.T.reshape(-1, 1))
		cov_landmark = (np.eye(3 * M) - K[:-6, :] @ H[:, :-6]) @ cov_landmark
		# transform the 3M x 1 back to 3 x M vector
		self.landmark[:, ind] = mu_landmark.reshape(-1, 3).T
		self.cov_landmark2landmark[cov_mask_x, cov_mask_y] = cov_landmark
		# use Kalman Gain to update the pose
		self.pose = scipy.linalg.expm(SE3_skew((K[-6:, :] @ (obs.T.reshape(-1, 1) - z.T.reshape(-1, 1))).flatten())) @ self.pose
		self.cov_pose2pose = (np.eye(6) - K[-6:, :] @ H[:, -6:]) @ self.cov_pose2pose

	def landmark_update(self, obs, ind_kpt):
		assert np.allclose(ind_kpt, np.intersect1d(np.where(self.observed==1)[0], ind_kpt))
		# transform landmark to 3M x 1 shape mean vector
		mean = self.landmark[:, ind_kpt].T.reshape(-1, 1)
		cov_mask = np.concatenate([ind_kpt * 3, ind_kpt * 3 + 1, ind_kpt * 3 + 2])
		cov_mask_x, cov_mask_y = np.meshgrid(cov_mask, cov_mask, sparse=False, indexing='xy')
		cov = self.cov_landmark2landmark[cov_mask_x, cov_mask_y]
		# obs: 4xN, mean: 1x3M, cov: 3Mx3M
		P = get_P()
		X = self.cam_T_imu @ self.pose @ homogenize(self.landmark[:, ind_kpt])
		# z: 4xN
		z_obs = self.M @ (X / X[2, :])
		# y: 4x4
		y = self.cam_T_imu @ self.pose @ P.T
		# H shape: 4Nx3M: obs x landmark
		H = np.zeros((4 * obs.shape[1], 3 * obs.shape[1]))
		for i in range(ind_kpt.shape[0]):
			# Hi: 4x3
			H[4*i:4*(i+1), 3*i:3*(i+1)] = self.M @ J_homo(X[:, i].reshape(-1,1)) @ y
		# H: 4N x 3M, N is the number of observation, M is the number of landmarks observed so far
		# K: 3M x 4N
		K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + np.eye(4 * obs.shape[1]) * landmark_var)
		# K @ (obs - z): 3M x 1
		mean = mean + K @ (obs.T.reshape(-1, 1) - z_obs.T.reshape(-1, 1))
		cov = (np.eye(K.shape[0]) - K @ H) @ cov
		# transform the 3M x 1 back to 3 x M vector
		self.landmark[:, ind_kpt] = mean.reshape(-1, 3).T
		self.cov_landmark2landmark[cov_mask_x, cov_mask_y] = cov

	def pose_update(self, obs, ind_kpt):
		"""
		EKF update the pose based on landmark observations
		"""
		N = obs.shape[1]
		X = self.pose @ homogenize(self.landmark[:, ind_kpt])
		z_obs = self.M @ ((self.cam_T_imu @ X) / (self.cam_T_imu @ X)[2, :])
		H = np.zeros((4 * N, 6))
		for i in range(N):
			H[4*i:4*(i+1), :] = self.M @ J_homo(self.cam_T_imu @ X[:, i].reshape(-1,1)) @ self.cam_T_imu @ get_homo_cur(X[:, i])
		K = self.cov_pose2pose @ H.T @ np.linalg.inv(H @ self.cov_pose2pose @ H.T + np.eye(4 * N) * sigma)
		self.pose = scipy.linalg.expm(SE3_skew((K @ (obs.T.reshape(-1, 1) - z_obs.T.reshape(-1, 1))).flatten())) @ self.pose
		self.cov_pose2pose = (np.eye(6) - K @ H) @ self.cov_pose2pose

	def landmark_init(self, obs):
		"""
		2D observations of the 3D landmark
		:param: observation - 4 x N
		:return: 4 x N
		"""
		landmarks_cam = stereo_pts_init(self.M, obs, self.cam_T_imu)
		return dehomogenize(self.get_pose_in_world() @ landmarks_cam)

	def record_pose(self, pose):
		# pose inversion
		self.poses.append(np.linalg.inv(pose))
	
	def get_poses(self):
		return self.poses

	def get_landmarks(self):
		return self.landmark

	def record_robustness(self, x):
		self.robustness.append(x)