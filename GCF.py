import cv2
import numpy as np
import matplotlib.pyplot as plt

# GCF based Feature Extraction

def min_projection(im): # Minimum Projection Operator
	r = im.shape[0]
	c = im.shape[1]
	im = im.astype(int)
	dm = np.zeros(im.shape)

	for i in range(0, r):
		for j in range(0, c):
			# d = np.full(8, 255) # for grayscale
			d = np.full((8,3), 255) # for color
			if i > 0 and i < r - 1:
				d[0] = (im[i - 1][j] + im[i + 1][j])/2 - im[i][j]
			if j > 0 and j < c - 1:
				d[1] = (im[i][j - 1] + im[i][j + 1])/2 - im[i][j]
			if i > 0 and i < r - 1 and  j > 0 and j < c - 1:
				d[2] = (im[i - 1][j - 1] + im[i + 1][j +1])/2 - im[i][j]
			if i > 0 and i < r - 1 and  j > 0 and j < c - 1:
				d[3] = (im[i - 1][j + 1] + im[i + 1][j - 1])/2 - im[i][j]
			if i > 0 and  j > 0:
				d[4] = im[i - 1][j] + im[i][j - 1] - im[i - 1][j -1] - im[i][j]
			if i > 0 and j < c - 1:
				d[5] = im[i - 1][j] + im[i][j + 1] - im[i - 1][j +1] - im[i][j]
			if i < r - 1 and  j > 0:
				d[6] = im[i + 1][j] + im[i][j - 1] - im[i + 1][j - 1] - im[i][j]
			if i < r - 1 and j < c - 1:
				d[7] = im[i + 1][j] + im[i][j + 1] - im[i + 1][j + 1] - im[i][j]

			dm[i][j] = np.min(d)
	return dm

def GC(im, m): # To apply Gaussian Curvature filter m times on image
	for i in range(0, m):
		im = im + min_projection(im)
	return im


# Focus Region Confirmation

def RF(f, p, q): # Row Frequency
	f = f.astype(float)
	rf = np.zeros(f.shape)

	for i in range(0, f.shape[0]):
		for j in range(0, f.shape[1]):
			count = 0.0
			r = 0
			for x in range(max(i - p/2, 0), min(i + p/2 + 1, f.shape[0])): # p should be odd
				for y in range(max(j - q/2 + 1, 0), min(j + p/2 + 1, f.shape[1])): # q should be odd to make patch of p*q
 					count = count + 1
 					r = r + (f[x][y] - f[x][y-1])**2
 			rf[i][j] = (r/count)**0.5
 	return rf

def CF(f, p, q): # Column Frequency
	f = f.astype(float)
	cf = np.zeros(f.shape)

	for i in range(0, f.shape[0]):
		for j in range(0, f.shape[1]):
			count = 0.0
			c = 0
			for x in range(max(i - p/2 + 1, 0), min(i + p/2 + 1, f.shape[0])): # p should be odd
				for y in range(max(j - q/2, 0), min(j + p/2 + 1, f.shape[1])): # q should be odd to make patch of p*q
 					count = count + 1
 					c = c + (f[x][y] - f[x - 1][y])**2
 			cf[i][j] = (c/count)**0.5
 	return cf

def SF(f, p, q): #Spatial Frequency
	rf = RF(f, p, q)
	cf = CF(f, p, q)
	return np.sqrt(rf * rf + cf * cf)

def LV(f, p, q): # Local Variance
	f = f.astype(float)
	lv = np.zeros(f.shape)
	u = np.mean(f)

	for i in range(0, f.shape[0]):
		for j in range(0, f.shape[1]):
			count = 0.0
			l = 0
			for x in range(max(i - p/2 + 1, 0), min(i + p/2 + 1, f.shape[0])): # p should be odd
				for y in range(max(j - q/2, 0), min(j + p/2 + 1, f.shape[1])): # q should be odd to make patch of p*q
 					count = count + 1
 					l = l + (f[x][y] - u)**2
 			lv[i][j] = l/count
 	return lv

def focus_maps(f1, f2, p, q): # Generates Focus Maps
	sf1 = SF(f1, p, q)
	sf2 = SF(f2, p, q)
	lv1 = LV(f1, p, q)
	lv2 = LV(f2, p, q)

	c1 = (np.greater(sf1, sf2).astype(int) + np.greater(lv1, lv2).astype(int))/2
	c2 = (np.greater(sf2, sf1).astype(int) + np.greater(lv2, lv1).astype(int))/2
	return c1, c2

# Filter Processing
def dilate(c, n):
	cm = np.zeros(c.shape)
	for i in range(0, c.shape[0]):
		for j in range(0, c.shape[1]):
			d = c[i][j]
			for x in range(max(i - n/2, 0), min(i + n/2 + 1, c.shape[0])): 
				for y in range(max(j - n/2, 0), min(j + n/2 + 1, c.shape[1])): # n should be odd to make patch of n*n
					d = np.maximum(c[x][y], d)
			cm[i][j] = d
	return cm

def erode(c, n):
	cm = np.zeros(c.shape)
	for i in range(0, c.shape[0]):
		for j in range(0, c.shape[1]):
			d = c[i][j]
			for x in range(max(i - n/2, 0), min(i + n/2 + 1, c.shape[0])): 
				for y in range(max(j - n/2, 0), min(j + n/2 + 1, c.shape[1])): # n should be odd to make patch of n*n
					d = np.minimum(c[x][y], d)
			cm[i][j] = d
	return cm

def morphological_transform(c, n):
	cm = dilate(c, n)
	cm = erode(cm, n)
	cn = erode(cm, n)
	cn = dilate(cn, n) 
	return cn

def median_filter(c, n):
	m = np.zeros(c.shape)

	for i in range(0, c.shape[0]):
		for j in range(0, c.shape[1]):
			count = 0
			members = []
			for x in range(max(i - n/2, 0), min(i + n/2 + 1, c.shape[0])): 
				for y in range(max(j - n/2, 0), min(j + n/2 + 1, c.shape[1])):
					count =count + 1
					members.append(c[i][j])
			np.sort(members)
			m[x][y] = members[count/2]

	return m

# im1 = cv2.imread("TestingImageDataset/testna_slika2a.bmp", cv2.IMREAD_GRAYSCALE)
# im2 = cv2.imread("TestingImageDataset/testna_slika2b.bmp", cv2.IMREAD_GRAYSCALE)
im1 = cv2.imread("TestingImageDataset/p30a.jpg")
im2 = cv2.imread("TestingImageDataset/p30b.jpg")

# Parameters
m =1 # no. of times gc filter is applied
p, q = 7, 7 # dimension of patch in focus region
n = 5 # dilation and erosion kernel size nxn

igc1 = GC(im1, m)
igc2 = GC(im2, m)

f1 = im1 - igc1
f2 = im2 - igc2

c1, c2 = focus_maps(f1, f2, p, q)

cn1 = morphological_transform(c1, n)
cn2 = morphological_transform(c2, n)

cm1 = median_filter(cn1, n)
cm2 = median_filter(cn2, n)

final1 = cm1*im1 + (1 - cm1)*im2
final2 = (1 - cm2)*im1 + cm2*im2


row = 2
column = 7
fig=plt.figure()
fig.add_subplot(row, column, 1)
plt.imshow(im1.astype(int), cmap='gray')

fig.add_subplot(row, column, 2)
plt.imshow(igc1.astype(int), cmap='gray')

fig.add_subplot(row, column, 3)
plt.imshow(f1.astype(int), cmap='gray')

fig.add_subplot(row, column, 4)
plt.imshow(c1.astype(int), cmap='gray')

fig.add_subplot(row, column, 5)
plt.imshow(cn1.astype(int), cmap='gray')

fig.add_subplot(row, column, 6)
plt.imshow(cm1.astype(int), cmap='gray')

fig.add_subplot(row, column, 7)
plt.imshow(final1.astype(int), cmap='gray')

fig.add_subplot(row, column, 8)
plt.imshow(im2.astype(int), cmap='gray')

fig.add_subplot(row, column, 9)
plt.imshow(igc2.astype(int), cmap='gray')

fig.add_subplot(row, column, 10)
plt.imshow(f2.astype(int), cmap='gray')

fig.add_subplot(row, column, 11)
plt.imshow(c2.astype(int), cmap='gray')

fig.add_subplot(row, column, 12)
plt.imshow(cn2.astype(int), cmap='gray')

fig.add_subplot(row, column, 13)
plt.imshow(cm2.astype(int), cmap='gray')

fig.add_subplot(row, column, 14)
plt.imshow(final2.astype(int), cmap='gray')

plt.show()

