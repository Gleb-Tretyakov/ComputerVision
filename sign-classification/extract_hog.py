from skimage import io, color, feature, transform
import numpy as np
import math
from numpy import ones

def get_derivate_image(img):
	arr = np.array([-1, 0, 1])
	arrT = arr.T
	height, weight = img.shape[0], img.shape[1]
	Ix = np.zeros((height, weight + 2))
	for i in range(height):
		Ix[i,:] = np.convolve(img[i,:], arr)
	Iy = np.zeros((height, weight + 2))
	for i in range(img.shape[0]):
		Iy[i,:] = np.convolve(img[i,:], arrT)
	return Ix, Iy
	
def get_gradient_angle(Ix, Iy):
	G = (Ix * Ix + Iy * Iy) ** 0.5
	alpha = (np.arctan2(Iy, Ix) ** 2) ** 0.5
	return G, alpha

def nonamefun1(gradient_matrix, angle_matrix, cellRows, cellCols):
	height, weight = gradient_matrix.shape
	size1 = np.int(height / cellRows)
	size2 = np.int(weight / cellCols)
	now_index = 0
	index_block = np.zeros((height, weight))
	index_block = index_block.astype(np.int)
	small_blocks = np.zeros((9))
	counter = 0
	for i in range(size1):
		for j in range(size2):
			current_block = np.zeros((9))
			for x in range(i * cellRows, (i + 1) * cellRows):
				for y in range(j * cellCols, (j + 1) * cellCols):
					index_block[x, y] = counter
					current_angle = angle_matrix[x, y]
					cur_ind = np.int(current_angle / np.pi * 9.0)
					current_block[cur_ind] += gradient_matrix[x, y]
			small_blocks = np.vstack((small_blocks, current_block))
			counter += 1
	snall_blocks = np.delete(small_blocks, 0, 0)
	return index_block, small_blocks

def nonamefun2(index_block, small_blocks, blockRowCells, blockColCells):
	height, weight = index_block.shape
	size1 = np.int(height / blockRowCells)
	size2 = np.int(weight / blockColCells)
	big_blocks = np.zeros((9))
	const_eps = 10 ** (-12)
	for i in range(size1):
		for j in range(size2):
			current_block = np.zeros((9))
			unique_index = set()
			x_begin = i * blockRowCells
			x_end = (i + 1) * blockRowCells
			y_begin = j * blockColCells
			y_end = (j + 1) * blockColCells
			if (x_end > height):
				x_end = height
				x_begin = height - blockRowCells
			if (y_end > weight):
				y_end = weight
				y_begin = weight - blockColCells
			for x in range(x_begin, x_end):
				for y in range(y_begin, y_end):
					unique_index.add(index_block[x, y])
			for ind in unique_index:
				current_block = np.hstack((current_block, small_blocks[ind]))
			current_block = np.delete(current_block, current_block[:9])
			current_block = current_block / np.sqrt(current_block.shape[0] ** 2 + const_eps)
			big_blocks = np.append(big_blocks, current_block)
	np.delete(big_blocks, 0, 0)
	return big_blocks



def extract_hog(img, roi):
	roi = roi.astype(np.int)
	img = img[roi[0]:roi[2], roi[1]:roi[3]]
	img = color.rgb2gray(img)
	now = 30.0
	img = transform.rescale(img, (now / img.shape[0], now / img.shape[1]))
	cellRows, cellCols = (4, 4)
	blockRowCells, blockColCells = (6, 6)
	Ix, Iy = get_derivate_image(img)
	gradient_matrix, angle_matrix = get_gradient_angle(Ix, Iy)
	index_block, small_blocks = nonamefun1(gradient_matrix, angle_matrix, cellRows, cellCols)
	big_blocks = nonamefun2(index_block, small_blocks, blockRowCells, blockColCells)
	return big_blocks
