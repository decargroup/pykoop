"""Benchmark :func:`pykoop.koopman_pipeline._unique_episodes()`."""

import timeit

import numpy as np

from pykoop.koopman_pipeline import _unique_episodes


def main():
    """Benchmark :func:`pykoop.koopman_pipeline._unique_episodes()`."""
    X_ep = np.array([0] * 100 + [1] * 1000 + [2] * 500 + [3] * 1000)
    n_loop = 100_000
    time = timeit.timeit(lambda: _unique_episodes(X_ep), number=n_loop)
    print(f'   Total time: {time} s')
    print(f'Time per loop: {time / n_loop} s')


if __name__ == '__main__':
    main()
