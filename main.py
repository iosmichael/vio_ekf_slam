import numpy as np
from tqdm import tqdm
from utils import *
from ekf_vio import EKF_VIO

config = {
	'dirname': 'data',
	'dataset': '0022.npz',
	'save_dirname':'expr_22'
}

if __name__ == '__main__':
	filename = f"./{config['dirname']}/{config['dataset']}"

	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)
	print('======== Dataset Overview ========\n')
	print(f"timestamps: {t.shape}")
	print(f"features: {features.shape}")
	print(f"linear_velocity: {linear_velocity.shape}")
	print(f"rotational_velocity: {rotational_velocity.shape}")
	print(f"K: {K}")
	print(f"stereo baseline: {b}")
	# from IMU to Left Camera Frame
	print(f"cam_T_imu: {cam_T_imu}")
	print('======== ================ ========\n')

	VIO_SLAM = EKF_VIO(config, K, b, cam_T_imu, features.shape[1])
	for i in tqdm(range(1, t.shape[1])):
		tau = np.abs(t[0, i] - t[0, i-1])
		linear, angular = linear_velocity[:, i], rotational_velocity[:, i]
		VIO_SLAM.advance_one_step(tau, (linear, angular), features[:, :, i])
		
		if i % 100 == 0:
			world_T_imu = np.stack(VIO_SLAM.get_poses(), axis=2)
			# print(world_T_imu.shape)
			# visualize_trajectory_2d(f"{config['save_dirname']}/VIO_{i}.png", world_T_imu, VIO_SLAM.get_landmarks(),show_ori=False)
	
	# You can use the function below to visualize the robot pose over time
	# world_T_imu = np.stack(VIO_SLAM.get_poses(), axis=2)
	# print(world_T_imu.shape)
	# visualize_trajectory_2d(f"{config['save_dirname']}/VIO_final.png", world_T_imu, VIO_SLAM.get_landmarks(),show_ori=False)
