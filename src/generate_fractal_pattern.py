import numpy as np

def generate_pattern(p_0, m, Δ, rng=None):
    assert len(s)==m
    if rng is None:
        rng = np.random.default_rng()
    tile = np.array((-.75, .25, .25, .25))
    pattern_init = np.array([[p_0]])
    signs_init = np.ones((1, 1))
    patterns = [pattern_init]
    signs = [signs_init]
    for scale in range(1, m+1):  # scale 0 is the initialized value
        new_signs = tile[:, None, None]*signs[-1][None]
        new_signs = rng.permuted(new_signs, axis=0)
        new_signs_0 = np.concatenate(new_signs[:2], axis=0)
        new_signs_1 = np.concatenate(new_signs[2:], axis=0)
        new_signs = np.concatenate((new_signs_0, new_signs_1), axis=1)
        



