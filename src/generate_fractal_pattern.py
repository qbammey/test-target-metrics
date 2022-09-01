import imageio
import numpy as np


def generate_fractal_pattern(Δ, p0=.5, init_yinyang=True, rng=None, return_all_scales=False):
    if rng is None:
        rng = np.random.default_rng()
    tile = np.array((-.75, .25, .25, .25))
    pattern_init = np.array([[p0]])
    signs_init = np.ones((1, 1))
    patterns = [pattern_init]
    signs = [signs_init]
    for scale, 𝛿 in enumerate(Δ):
        if scale == 0 and init_yinyang:
            increment = np.array((-.5, .5, .5, -.5))[:, None, None] * signs[-1][None]
            if rng.random() < .5:
                increment *= -1
        else:
            increment = tile[:, None, None] * signs[-1][None]
            increment = rng.permuted(increment, axis=0)
        new_pattern = patterns[-1].repeat(2, axis=0).repeat(2, axis=1)
        upscaled = np.zeros_like(new_pattern)
        upscaled[::2, ::2] = increment[0]
        upscaled[1::2, ::2] = increment[1]
        upscaled[::2, 1::2] = increment[2]
        upscaled[1::2, 1::2] = increment[3]
        new_pattern += upscaled * 𝛿
        # increment_0 = np.concatenate(increment[:2], axis=0)
        # increment_1 = np.concatenate(increment[2:], axis=0)
        # increment = np.concatenate((increment_0, increment_1), axis=1)
        # new_pattern = patterns[-1].repeat(2, axis=0).repeat(2, axis=1) + increment * 𝛿
        new_sign = np.sign(upscaled)
        patterns.append(new_pattern)
        signs.append(new_sign)
    if return_all_scales:
        return patterns[-1].clip(0, 1), patterns, signs
    else:
        return patterns[-1].clip(0, 1)


def generate_constant_contrast_target(n=7, 𝛿=.15, 𝛿0=.60, **kwargs):
    assert n >= 3
    Δ = [𝛿0] * 2 + [𝛿] * (n - 2)
    return generate_fractal_pattern(Δ, **kwargs)


def generate_variable_contrast_target_v1(n=7, 𝛿0=.60, k=np.sqrt(2), **kwargs):
    Δ = [𝛿0] * 2
    for i in range(n - 2):
        Δ.append(Δ[-1] / k)
    return generate_fractal_pattern(Δ, **kwargs)


if __name__ == "__main__":
    from sys import argv
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_scales", default=7, type=int)
    parser.add_argument("-t", "--type", type=str, choices=["constant", "variable"], default="variable")
    parser.add_argument("-c", "--contrast-init", type=float, default=.6)
    parser.add_argument("-i", "--contrast-increment", type=float, default=.15)
    parser.add_argument("-k", "--contrast-divider", type=float, default=np.sqrt(2))
    parser.add_argument("-f", "--first-tile", type=str, choices=["yinyang", "corner"], default="yinyang")
    parser.add_argument("-p", "--initial-average", type=float, default=.5)
    args = parser.parse_args(argv[1:])
    init_yinyang = args.first_tile == 'yinyang'
    if args.type == "constant":
        pattern, all_patterns, _ = generate_constant_contrast_target(args.n_scales, args.contrast_init,
                                                                     args.contrast_increment, p0=args.initial_average,
                                                                     init_yinyang=init_yinyang, return_all_scales=True)
    else:
        pattern, all_patterns, _ = generate_variable_contrast_target_v1(args.n_scales, args.contrast_init,
                                                                        args.contrast_divider, p0=args.initial_average,
                                                                        init_yinyang=init_yinyang,
                                                                        return_all_scales=True)
    imageio.imsave("out.png", np.round(pattern * 255).astype(np.uint8))
    Ye, Xe = pattern.shape
    for i, p in enumerate(all_patterns):
        Y, X = p.shape
        rY, rX = Ye//Y, Xe//X
        p = p.repeat(rY, axis=0).repeat(rX, axis=1).clip(0, 1)
        imageio.imsave(f"out_{i}.png", np.round(p * 255).astype(np.uint8))
