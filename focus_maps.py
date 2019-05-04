import numpy as np

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
            cf[i][j] = (c/count) ** 0.5
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

def focus_maps(f1, f2, p, q, color=False): # Generates Focus Maps
    sf1 = SF(f1, p, q)
    sf2 = SF(f2, p, q)
    lv1 = LV(f1, p, q)
    lv2 = LV(f2, p, q)

    if color:
        sf1 = np.sum(sf1, axis=2)
        sf2 = np.sum(sf2, axis=2)
        lv1 = np.sum(lv1, axis=2)
        lv2 = np.sum(lv2, axis=2)

    c1 = (np.greater(sf1, sf2).astype(int) + np.greater(lv1, lv2).astype(int))/2
    c2 = (np.greater(sf2, sf1).astype(int) + np.greater(lv2, lv1).astype(int))/2
    return c1, c2
