import numpy as np
import scipy as sp
import scipy.ndimage

from skimage.util import view_as_blocks
import imageio  # debug
# ref = imageio.imread("example_target.png") / 255
# obs = imageio.imread("example_target_degraded.png") / 255

def downsample_block(a, σ=None):
    if σ is not None and σ > 0:
        a = sp.ndimage.gaussian_filter(a, σ)
    return view_as_blocks(a, (2, 2)).mean(axis=(-1, -2))


def pcr_one_scale(obs, ref, thr=None):
    ref_block = view_as_blocks(ref, (2, 2))
    Y, X, _, _ = ref_block.shape
    ref_block = ref_block.reshape(Y, X, 4)
    n_comparisons = Y * X
    ref_med = np.median(ref_block, axis=-1)
    ref_difftomed = ref_block - ref_med[:, :, None]
    ref_strongest = np.argmax(np.abs(ref_difftomed), axis=-1)
    mesh = np.meshgrid(range(ref_strongest.shape[0]), range(ref_strongest.shape[1]))
    ref_contrast = ref_difftomed[mesh[0], mesh[1], ref_strongest]
    ref_sign = np.sign(ref_contrast)
    obs_block = view_as_blocks(obs, (2, 2)).reshape(Y, X, 4)
    obs_med = np.median(obs_block, axis=-1)
    obs_difftomed = obs_block - obs_med[:, :, None]
    obs_strongest = np.argmax(np.abs(obs_difftomed), axis=-1)
    mesh = np.meshgrid(range(obs_strongest.shape[0]), range(obs_strongest.shape[1]))
    obs_strongest = np.argmax(np.abs(obs_difftomed), axis=-1)
    obs_contrast = obs_difftomed[mesh[0], mesh[1], obs_strongest]
    obs_sign = np.sign(obs_contrast)
    correct_sign = ref_sign == obs_sign
    correct_strongest = ref_strongest == obs_strongest
    pass_threshold = np.abs(obs_contrast)>= thr if thr is not None else np.ones(correct_sign.shape, bool)
    correlation = 3 * correct_sign*correct_strongest +\
        1 * (~correct_sign) * (~correct_strongest) +\
        -1 * correct_sign * (~correct_strongest) +\
        -3 * (~correct_sign) * correct_strongest
    # n_pass_threshold = np.count_nonzero(pass_threshold)
    pcr = 1 / (3 * n_comparisons) * np.sum(correlation * pass_threshold)
    return pcr

def compute_pcr(obs, ref, σ=None, thr=None):
    assert obs.shape == ref.shape
    pcrs = []
    while min(*obs.shape) >= 2:
        pcr = pcr_one_scale(obs, ref, thr)
        pcrs.append(pcr)
        obs = downsample_block(obs, σ)
        ref = downsample_block(ref, σ)
    return pcrs


if False and __name__ == "__main__":
    import argparse
    from sys import argv

    parser = argparse.ArgumentParser()
    parser.add_argument("obs", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("-s", "--sigma", type=float, default=None)
    parser.add_argument("-t", "--threshold", type=float, default=None)
    args = parser.parse_args(argv[1:])
    obs = imageio.imread(args.obs) / 255
    ref = imageio.imread(args.ref) / 255
    pcrs = compute_pcr(obs, ref, σ=args.σ, thr=args.thr)
    for pcr in pcrs:
        print(pcr)