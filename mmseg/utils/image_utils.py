
import cv2
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from torch import Tensor


def save_cv2_image_as_chinese_path(img, dst_path, is_bgr=False):
	""" using cv2 to saving image in chinese path

	Args:
		img (ndarray): image
		dst_path (str): destination path
		is_bgr (bool): if the channel order of image is BGR. Default: False
	"""
	if not is_bgr:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	ext_name = osp.splitext(dst_path)[1]
	cv2.imencode(ext_name, img)[1].tofile(dst_path)


def read_cv2_image_as_chinese_path(img_path, dtype=np.uint8):
	""" using cv2 to read image in chinese path

	Args:
		img_path (str): image path
		dtype (np.dtype): image save data type. Default: np.uint8
	"""
	return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def save_image_by_cv2(img, dst_path, is_bgr=False, if_norm=True):
	""" Save image by cv2.imwrite, this function automatic transforms the data range and data type to adapt to cv2

	Args:
		img (ndarray): image to be saved
		dst_path (str): save path
		is_bgr (bool): if the channel order of image is BGR. Default: False
		if_norm (bool): whether to noralize to [0, 1]. Default: True

	Returns:
		True if succeed, False otherwise
	"""
	if isinstance(img, Tensor):
		img = img.numpy()
		
	if img.dtype == np.uint8:
		new_img = img
	
	elif img.dtype in (np.float32, np.float64):
		
		# add a new axis for grayscale image
		if img.ndim==2:
			img = img[:, :, np.newaxis]

		new_img = np.empty_like(img, dtype=np.uint8)

		for ii in range(img.shape[2]):
			sub_img = img[..., ii]
			if if_norm:
				sub_img = min_max_map(sub_img)
			sub_img = (255*sub_img).astype(np.uint8)
			new_img[..., ii] = sub_img
			
	elif img.dtype == np.int64:
		new_img = img.astype(np.uint8)

	new_img = new_img.squeeze()
	return save_cv2_image_as_chinese_path(new_img, dst_path, is_bgr=is_bgr)


def plot_surface(img, cmap='jet'):
	""" plot 3D surface of image
	"""

	h, w = img.shape

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	X = np.arange(h)
	Y = np.arange(w)
	X, Y = np.meshgrid(X, Y)

	surf = ax.plot_surface(X, Y, img, cmap=cmap)

	return fig