import time
import numpy as np

def loop_distances(X):
    """Pairwise Euclidean distances via nested loops.
    X: (N, d) array. Returns D: (N, N) array with D[i, j] = ||x_i - x_j||.
    """
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = X[i] - X[j]
            D[i, j] = np.sqrt(np.sum(diff * diff))
    return D

def broadcast_distances(X):
    """Pairwise Euclidean distances via NumPy broadcasting, no loops.
    X: (N, d) array. Returns D: (N, N) array with D[i, j] = ||x_i - x_j||.
    """
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))
    return D

def expansion_distances(X):
    """Pairwise Euclidean distances via ||xi||^2 + ||xj||^2 - 2 xi.xj.
    X: (N, d) array. Returns D: (N, N) array with D[i, j] = ||x_i - x_j||.
    Clips small negative values before sqrt (roundoff cancellation).
    """
    sq_norms = np.sum(X * X, axis=1)
    gram = X @ X.T
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * gram
    sq_dists = np.maximum(0.0, sq_dists)
    D = np.sqrt(sq_dists)
    return D

def peak_memory_broadcast_bytes(N, d, itemsize=8):
    return N * N * d * itemsize

def _time_fn(fn, X, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

def main():
    rng = np.random.default_rng(0)
    N, d = 2000, 20
    X = rng.standard_normal((N, d))

    N_check = 200
    X_check = X[:N_check]
    D_loop = loop_distances(X_check)
    D_bcast = broadcast_distances(X_check)
    D_exp = expansion_distances(X_check)

    agree_loop_bcast = np.allclose(D_loop, D_bcast, atol=1e-10)
    max_diff_exp = np.max(np.abs(D_loop - D_exp))
    agree_loop_exp_1e10 = np.allclose(D_loop, D_exp, atol=1e-10)
    agree_loop_exp_1e6 = np.allclose(D_loop, D_exp, atol=1e-6)
    print(f"loop vs broadcast allclose (atol=1e-10): {agree_loop_bcast}")
    print(f"loop vs expansion max abs diff: {max_diff_exp:.3e}")
    print(f"loop vs expansion allclose (atol=1e-10): {agree_loop_exp_1e10}")
    print(f"loop vs expansion allclose (atol=1e-6):  {agree_loop_exp_1e6}")
    print("NOTE: broadcasting matches the loop method to 1e-10 (both compute")
    print("(xi - xj) directly). The expansion trick computes ||xi||^2 + ||xj||^2")
    print("- 2 xi.xj, which subtracts two close, similarly-sized floating point")
    print("numbers -> catastrophic cancellation -> agreement only to ~1e-7,")
    print("not 1e-10. This IS the numerical-stability cost referred to in 1(c).")

    N_loop_time = 300
    X_loop_time = X[:N_loop_time]
    t_loop = _time_fn(loop_distances, X_loop_time, repeats=1)
    t_loop_scaled = t_loop * (N / N_loop_time) ** 2

    t_bcast = _time_fn(broadcast_distances, X, repeats=5)
    t_exp = _time_fn(expansion_distances, X, repeats=5)

    mem_bcast_mb = peak_memory_broadcast_bytes(N, d) / 1e6
    mem_exp_mb = (N * N * 8) / 1e6

    print("\n--- Timing table (N=2000, d=20) ---")
    print(f"{'Method':<28}{'Time (s)':<14}{'Peak mem (MB)':<16}{'Speedup vs loop':<16}")
    print(f"{'Loop (measured@N=300, scaled)':<28}{t_loop_scaled:<14.4f}{'~O(N) small':<16}{'1.0x (ref)':<16}")
    print(f"{'Broadcasting':<28}{t_bcast:<14.6f}{mem_bcast_mb:<16.1f}{t_loop_scaled / t_bcast:<16.1f}")
    print(f"{'Expansion trick':<28}{t_exp:<14.6f}{mem_exp_mb:<16.1f}{t_loop_scaled / t_exp:<16.1f}")

    with open("results/q1_timing.txt", "w") as f:
        f.write("Q1(c) Distance matrix implementations: correctness and timing\n")
        f.write("=" * 65 + "\n")
        f.write(f"loop vs broadcast allclose (atol=1e-10): {agree_loop_bcast}\n")
        f.write(f"loop vs expansion max abs diff: {max_diff_exp:.3e}\n")
        f.write(f"loop vs expansion allclose (atol=1e-10): {agree_loop_exp_1e10}\n")
        f.write(f"loop vs expansion allclose (atol=1e-6):  {agree_loop_exp_1e6}\n\n")
        f.write(f"N={N}, d={d}\n\n")
        f.write(f"{'Method':<28}{'Time (s)':<14}{'Peak mem (MB)':<16}{'Speedup vs loop':<16}\n")
        f.write(f"{'Loop (scaled from N=300)':<28}{t_loop_scaled:<14.4f}{'small':<16}{'1.0x (ref)':<16}\n")
        f.write(f"{'Broadcasting':<28}{t_bcast:<14.6f}{mem_bcast_mb:<16.1f}{t_loop_scaled / t_bcast:<16.1f}\n")
        f.write(f"{'Expansion trick':<28}{t_exp:<14.6f}{mem_exp_mb:<16.1f}{t_loop_scaled / t_exp:<16.1f}\n")
    print("\nSaved results/q1_timing.txt")


if __name__ == "__main__":
    main()
