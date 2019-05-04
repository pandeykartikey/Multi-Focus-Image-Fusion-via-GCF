import numpy as np

# GCF based Feature Extraction

def min_projection(im, color=False): # Minimum Projection Operator
    r = im.shape[0]
    c = im.shape[1]
    im = im.astype(int)
    dm = np.zeros(im.shape)

    for i in range(0, r):
        for j in range(0, c):

            if color:
                d = np.full((8,3), 255) # for color
            else:
                d = np.full(8, 255) # for grayscale

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

            if color:
                dm[i][j] = np.min(d, axis=0)
            else:
                dm[i][j] = np.min(d)

    return dm

def GC(im, m, color=False): # To apply Gaussian Curvature filter m times on image
    for i in range(0, m):
        im = im + min_projection(im, color)
    return im
