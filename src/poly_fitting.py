import numpy as np
import matplotlib.pyplot as plt

SIGMA = 0.3
SEED = 42

def true_function(x):
    return np.sin(2 * np.pi * x)

def generate_data(n, rng, sigma=SIGMA):
    x = rng.uniform(0, 1, size=n)
    y = true_function(x) + rng.normal(0, sigma, size=n)
    return x, y

def fit_poly(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return np.poly1d(coeffs)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def part_b(x_train, y_train, outpath):
    degrees = [0, 1, 3, 9]
    x_grid = np.linspace(0, 1, 500)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, M in zip(axes.ravel(), degrees):
        model = fit_poly(x_train, y_train, M)
        ax.scatter(x_train, y_train, color="tab:blue", s=25, label="Training data", zorder=3)
        ax.plot(x_grid, true_function(x_grid), color="tab:green", lw=2, label=r"True $f^\star(x)$")
        ax.plot(x_grid, model(x_grid), color="tab:red", lw=2, label=f"Fit (M={M})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Degree M = {M}")
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8)
    fig.suptitle("Polynomial fits of varying degree (n=15 training points)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def part_c_and_d(n, rng_seed, outpath, degrees=range(0, 13)):
    rng = np.random.default_rng(rng_seed)
    x_train, y_train = generate_data(n, rng)
    rng_test = np.random.default_rng(rng_seed + 1000)
    x_test, y_test = generate_data(1000, rng_test)

    train_rmses, test_rmses = [], []
    for M in degrees:
        model = fit_poly(x_train, y_train, M)
        train_rmses.append(rmse(y_train, model(x_train)))
        test_rmses.append(rmse(y_test, model(x_test)))

    train_rmses = np.array(train_rmses)
    test_rmses = np.array(test_rmses)
    degrees = np.array(list(degrees))

    best_test_M = degrees[np.argmin(test_rmses)]
    best_train_M = degrees[np.argmin(train_rmses)]
    gap = test_rmses - train_rmses
    largest_gap_M = degrees[np.argmax(gap)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(degrees, train_rmses, "o-", label="Train RMSE")
    ax.plot(degrees, test_rmses, "o-", label="Test RMSE")
    ax.set_xlabel("Polynomial degree M")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Train/Test RMSE vs. degree (n={n})")
    ax.set_yscale("log")
    ax.legend()
    ax.axvline(best_test_M, color="gray", ls="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    return {
        "n": n,
        "degrees": degrees,
        "train_rmses": train_rmses,
        "test_rmses": test_rmses,
        "best_test_M": int(best_test_M),
        "best_train_M": int(best_train_M),
        "largest_gap_M": int(largest_gap_M),
    }

def main():
    rng = np.random.default_rng(SEED)
    x_train15, y_train15 = generate_data(15, rng)

    part_b(x_train15, y_train15, "figures/q2b_fits_by_degree.png")

    res_n15 = part_c_and_d(15, SEED, "figures/q2c_rmse_vs_degree_n15.png")
    res_n100 = part_c_and_d(100, SEED, "figures/q2d_rmse_vs_degree_n100.png")

    with open("results/q2_summary.txt", "w") as f:
        f.write("Problem 2 summary\n")
        f.write("=" * 40 + "\n\n")
        for res in (res_n15, res_n100):
            f.write(f"n = {res['n']}\n")
            f.write(f"  M minimizing TEST RMSE : {res['best_test_M']} "
                    f"(test RMSE = {res['test_rmses'][res['best_test_M']]:.4f})\n")
            f.write(f"  M minimizing TRAIN RMSE: {res['best_train_M']} "
                    f"(train RMSE = {res['train_rmses'][res['best_train_M']]:.4f})\n")
            f.write(f"  M with largest generalization gap: {res['largest_gap_M']}\n")
            f.write("  Table (M, train_rmse, test_rmse):\n")
            for M, tr, te in zip(res["degrees"], res["train_rmses"], res["test_rmses"]):
                f.write(f"    {M:2d}  {tr:10.4f}  {te:10.4f}\n")
            f.write("\n")

    print("Problem 2 done. See figures/q2b_*, q2c_*, q2d_*, and results/q2_summary.txt")
    print(f"n=15:  best test M = {res_n15['best_test_M']}, "
          f"best train M = {res_n15['best_train_M']}, largest gap M = {res_n15['largest_gap_M']}")
    print(f"n=100: best test M = {res_n100['best_test_M']}, "
          f"best train M = {res_n100['best_train_M']}, largest gap M = {res_n100['largest_gap_M']}")

if __name__ == "__main__":
    main()
