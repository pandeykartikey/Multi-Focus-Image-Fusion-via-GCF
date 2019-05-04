import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from GCF import GC
from focus_maps import focus_maps
from morphological import morphological_transform
from median import median_filter

def parse_args():
    """
    Creates and returns the Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess your data for HAN model")
    parser.add_argument("--image1path", dest="image1path",
                    help="Path to the Image 1.",
                    default="./Dataset/testna_slika1a.bmp", type=str)
    parser.add_argument("--image2path", dest="image2path",
                    help="Path to the Image 2.",
                    default="./Dataset/testna_slika1b.bmp", type=str)
    parser.add_argument("--color", dest="color",
                    help="Whether the image is colored or not",
                    default=False, type=bool)
    parser.add_argument("-m", dest="m",
                    help="no. of times gaussian curvature filter is applied",
                    default=1, type=int)
    parser.add_argument("-n", dest="n",
                    help="dialation and erosion kernel size",
                    default=5, type=int)
    parser.add_argument("-p", dest="p",
                    help="pxq dimension of patch in focus region",
                    default=7, type=int)
    parser.add_argument("-q", dest="q",
                    help="pxq dimension of patch in focus region",
                    default=7, type=int)
    parser.add_argument("--outputdir", dest="outputdir",
                    help="Directory in which output needs to be stored",
                    default="./outputs/"+str(time.time()), type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)


    # Parameters
    color = args.color
    m = args.m # no. of times gc filter is applied
    p, q = args.p, args.q # dimension of patch in focus region
    n = args.n # dilation and erosion kernel size nxn

    if color:
        im1 = cv2.imread(args.image1path)
        im2 = cv2.imread(args.image2path)
    else :
        im1 = cv2.imread(args.image1path, cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(args.image2path, cv2.IMREAD_GRAYSCALE)

    igc1 = GC(im1, m, color)
    igc2 = GC(im2, m, color)

    f1 = im1 - igc1
    f2 = im2 - igc2

    c1, c2 = focus_maps(f1, f2, p, q, color)

    cn1 = morphological_transform(c1, n)
    cn2 = morphological_transform(c2, n)

    cm1 = median_filter(cn1, n)
    cm2 = median_filter(cn2, n)

    final1 = np.zeros(im1.shape)
    final2 = np.zeros(im2.shape)
    if color:
        for i in range(3):
            final1[:,:,i] = cm1*im1[:,:,i] + (1 - cm1)*im2[:,:,i]
            final2[:,:,i] = (1 - cm2)*im1[:,:,i] + cm2*im2[:,:,i]
    else:
        final1 = cm1*im1 + (1 - cm1)*im2
        final2 = (1 - cm2)*im1 + cm2*im2

    row = 2
    column = 7
    fig=plt.figure()
    fig.add_subplot(row, column, 1)
    plt.imshow(im1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/im1.png', im1.astype(int))

    fig.add_subplot(row, column, 2)
    plt.imshow(igc1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/igc1.png', igc1.astype(int))

    fig.add_subplot(row, column, 3)
    plt.imshow(f1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/f1.png', f1.astype(int))

    fig.add_subplot(row, column, 4)
    plt.imshow(c1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/c1.png', (c1*255).astype(int))

    fig.add_subplot(row, column, 5)
    plt.imshow(cn1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/cn1.png', (cn1*255).astype(int))

    fig.add_subplot(row, column, 6)
    plt.imshow(cm1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/cm1.png', (cm1*255).astype(int))

    fig.add_subplot(row, column, 7)
    plt.imshow(final1.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/final1.png', final1.astype(int))

    fig.add_subplot(row, column, 8)
    plt.imshow(im2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/im2.png', im2.astype(int))

    fig.add_subplot(row, column, 9)
    plt.imshow(igc2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/igc2.png', igc2.astype(int))

    fig.add_subplot(row, column, 10)
    plt.imshow(f2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/f2.png', f2.astype(int))

    fig.add_subplot(row, column, 11)
    plt.imshow(c2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/c2.png', (c2*255).astype(int))

    fig.add_subplot(row, column, 12)
    plt.imshow(cn2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/cn2.png', (cn2*255).astype(int))

    fig.add_subplot(row, column, 13)
    plt.imshow(cm2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/cm2.png', (cm2*255).astype(int))

    fig.add_subplot(row, column, 14)
    plt.imshow(final2.astype(int), cmap='gray')
    cv2.imwrite(args.outputdir+'/final2.png', final2.astype(int))

    plt.show()
