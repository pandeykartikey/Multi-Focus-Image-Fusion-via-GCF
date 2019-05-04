import numpy as np

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
