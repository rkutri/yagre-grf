import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import dct, dst, fft, next_fast_len


def benchmark_transform_times(sizes, n_trials=100):
    """
    Benchmark DCT-I (type=1, norm='backward') along axis=1
    for each grid size in `sizes`, averaging over `n_trials`.
    Returns a list of average times.
    """

    methods = ["fft", "dct", "dst"]
    times = ([], [], [])

    for i, method in enumerate(methods):

        if method == "fft":
            def transform(x): return fft(x, norm="backward", axis=1)
        elif method == "dct":
            def transform(x): return dct(x, type=1, norm="backward", axis=1)
        elif method == "dst":
            def transform(x): return dst(x, type=1, norm="backward", axis=1)

        for n in sizes:
            # Create random array of shape (n, n)
            arr = np.random.rand(n, n)

            # Warm‑up call to avoid one‑time planning costs
            _ = transform(arr)

            # Time multiple trials
            total_time = 0.0
            for _ in range(n_trials):
                start = time.perf_counter()
                _ = transform(arr)
                total_time += (time.perf_counter() - start)

            avg_time = total_time / n_trials
            times[i].append(avg_time)
            print(f"n={n:<4} → {avg_time:.4e} s per call")

    return times


if __name__ == "__main__":

    print("start")

    # Grid sizes to test (powers of 2)
    sizes = [16, 32, 64, 128, 257, 512, 1024, 2048]
    inputSizes = [n**2 for n in sizes]
    n_trials = 100

    # Run benchmark
    times_fft, times_dct, times_dst = benchmark_transform_times(
        sizes, n_trials)

    # Plot results
    plt.figure(figsize=(6, 4))
    plt.plot(inputSizes, times_fft, marker='o', label="fft")
    plt.plot(inputSizes, times_dct, marker='o', label="dct")
    plt.plot(inputSizes, times_dst, marker='o', label="dst")

    plt.xlabel('Input Size (n), total dofs')
    plt.ylabel('Avg Time per transform (s)')
    plt.title('Benchmark: Runtime vs Input Size')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()

    plt.savefig("transforms_comparison.pdf", format="pdf", dpi=300)

    plt.tight_layout()
    plt.show()
