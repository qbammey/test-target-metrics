import argparse
from sys import argv
from numba import njit
import numpy as np
import imageio
from skimage.util import view_as_blocks
import scipy as sp
import scipy.ndimage
from matplotlib import pyplot as plt
import seaborn as sns
from pcr import downsample_block

def get_contrast_histogram(img, max_dist=8, contrast_k=100, light_k=100):
    contrasts = np.linspace(0, 1, contrast_k)
    lights = np.linspace(0, 1, light_k)
    histogram = np.zeros((contrast_k, light_k), int)
    dists = ([(x, y) for x in range(max_dist+1) for y in range(max_dist+1) if (x or y) and (x**2 + y**2 <= max_dist**2)])
    _fill_histogram(img, histogram, dists, max_dist, contrast_k, light_k)
    return histogram

@njit
def _fill_histogram(img, histogram, dists, max_dist, contrast_k, light_k):
    Y, X = img.shape
    for y in range(Y):
        for x in range(X):
            intensity = img[y, x]
            for dy, dx in dists:
                yy, xx = y+dy, x+dx
                if yy>=Y or xx>=X:
                    continue
                v0 = img[y, x]
                v1 = img[y+dy, x+dx]
                intensity = (v0 + v1) / 2
                intensity_bin = int(np.round(intensity*light_k))
                contrast = np.abs(v0 - v1)
                contrast_bin = int(np.round(contrast*contrast_k))
                histogram[contrast_bin, intensity_bin] += 1
    return histogram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--contrast-bins", type=int, default=50)
    parser.add_argument("-i", "--intensity-bins", type=int, default=25)
    parser.add_argument("-d", "--distance", type=int, default=8)
    parser.add_argument("-s", "--downsample", type=float, default=1.4)
    parser.add_argument("image", type=str)
    args = parser.parse_args(argv[1:])
    img = imageio.imread(args.image) / 255
    if img.ndim == 3:
        img = img.mean(axis=-1)
    contrast_k = args.contrast_bins
    light_k = args.light_bins
    max_dist = args.distance
    σ = args.downsample if args.downsample>1e-10 else None

    scale = 0
    while min(*img.shape)>=8 and scale<8:
        hist = get_contrast_histogram(img, max_dist=max_dist, light_k=light_k, contrast_k=contrast_k)
        plt.matshow(hist, vmin=0, vmax=50, cmap="mako")
        plt.colorbar()
        plt.xlabel("Intensity")
        plt.ylabel("Contrast")
        plt.title(f"Scale {scale}")
        Y, X = img.shape
        img = img[:Y-Y%2, :X-X%2]
        img = downsample_block(img, σ)
        plt.savefig(f"s{scale}.png")
        plt.close()
        scale += 1




