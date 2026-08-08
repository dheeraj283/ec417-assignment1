import numpy as np
import matplotlib.pyplot as plt

try:
    from src.poly_fitting import generate_data, fit_poly, true_function, SIGMA
except ImportError:
    from poly_fitting import generate_data, fit_poly, true_function, SIGMA

SEED = 42
GRID = np.linspace(0, 1, 100)

def simulate_predictions(M, S, n, seed):
    rng = np.random.default_rng(seed)
    Fhat = np.zeros((S, len(GRID)))
    for s in range(S):
        x_train, y_train = generate_data(n, rng)
        model = fit_poly(x_train, y_train, M)
        Fhat[s] = model(GRID)
    return Fhat

def bias_var(Fhat):
    fbar = Fhat.mean(axis=0)                      
    bias2 = (fbar - true_function(GRID)) ** 2      
    var = Fhat.var(axis=0)                        
    return fbar, bias2, var

def part_b_figure(M_list, S, n, seed, outpath):
    fig, axes = plt.subplots(1, len(M_list), figsize=(5 * len(M_list), 4.5), sharey=True)
    for ax, M in zip(axes, M_list):
        rng = np.random.default_rng(seed)
        Fhat = np.zeros((S, len(GRID)))
        for s in range(S):
            x_train, y_train = generate_data(n, rng)
            model = fit_poly(x_train, y_train, M)
            Fhat[s] = model(GRID)
        fbar = Fhat.mean(axis=0)

        for s in range(20):
            ax.plot(GRID, Fhat[s], color="lightgray", lw=0.8, alpha=0.7, zorder=1)
        ax.plot(GRID, true_function(GRID), color="tab:green", lw=2.5,
                 label=r"True $f^\star$", zorder=3)
        ax.plot(GRID, fbar, color="tab:red", lw=2.5, ls="--",
                 label=r"Average prediction $\bar{f}$", zorder=3)
        ax.set_title(f"M = {M}" + ("  (y-axis clipped; M=9 is highly unstable)" if M == 9 else ""))
        ax.set_xlabel("x")
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("y")
    fig.suptitle(f"Bias-variance visualization: 20 fits + average, over S={S} datasets (n={n})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def part_c(M_list, S, n, seed):
    """Return dict M -> (mean_bias2, mean_var)."""
    out = {}
    for M in M_list:
        Fhat = simulate_predictions(M, S, n, seed)
        _, bias2, var = bias_var(Fhat)
        out[M] = (bias2.mean(), var.mean())
    return out

def part_e_identity_check(M_list, S, n, sigma, seed):
    """Verify Noise + Bias^2 + Var == E[(y - fhat)^2] via Monte Carlo."""
    rows = []
    for M in M_list:
        rng = np.random.default_rng(seed)
        Fhat = np.zeros((S, len(GRID)))
        rng_train = np.random.default_rng(seed)
        for s in range(S):
            x_train, y_train = generate_data(n, rng_train)
            model = fit_poly(x_train, y_train, M)
            Fhat[s] = model(GRID)
        fbar, bias2, var = bias_var(Fhat)

        rng_noise = np.random.default_rng(seed + 777)
        eps = rng_noise.normal(0, sigma, size=(S, len(GRID)))
        y_samples = true_function(GRID)[None, :] + eps          
        total_sq_err = (y_samples - Fhat) ** 2                  
        total_mc = total_sq_err.mean(axis=0)                    

        avg_total_mc = total_mc.mean()
        avg_bias2 = bias2.mean()
        avg_var = var.mean()
        rhs = sigma ** 2 + avg_bias2 + avg_var
        rows.append({
            "M": M, "sigma2": sigma ** 2, "bias2": avg_bias2, "var": avg_var,
            "sum": rhs, "simulated_total": avg_total_mc,
            "discrepancy": abs(rhs - avg_total_mc),
        })
    return rows

def part_f(M_list, S, n, seed, outpath):
    bias2_list, var_list, sum_list = [], [], []
    for M in M_list:
        Fhat = simulate_predictions(M, S, n, seed)
        _, bias2, var = bias_var(Fhat)
        bias2_list.append(bias2.mean())
        var_list.append(var.mean())
        sum_list.append(bias2.mean() + var.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(M_list, bias2_list, "o-", label=r"Bias$^2$")
    ax.plot(M_list, var_list, "o-", label="Variance")
    ax.plot(M_list, sum_list, "o-", label=r"Bias$^2$ + Variance")
    ax.set_xlabel("Polynomial degree M")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.set_title("Bias$^2$ / Variance vs. model complexity (n=15)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return bias2_list, var_list, sum_list

def main():
    n = 15
    M_list_bc = [1, 3, 9]
    M_list_full = list(range(0, 13))

    part_b_figure(M_list_bc, S=200, n=n, seed=SEED, outpath="figures/q3b_fits_and_average.png")

    bc_result = part_c(M_list_bc, S=200, n=n, seed=SEED)
    with open("results/q3c_bias_var.txt", "w") as f:
        f.write("Problem 3(c): average Bias^2 and Variance over the grid\n")
        f.write(f"{'M':<5}{'Bias^2':<12}{'Var':<12}\n")
        for M, (b2, v) in bc_result.items():
            f.write(f"{M:<5}{b2:<12.5f}{v:<12.5f}\n")
    print("3(c):", bc_result)

    rows_s200 = part_e_identity_check(M_list_bc, S=200, n=n, sigma=SIGMA, seed=SEED)
    rows_s2000 = part_e_identity_check(M_list_bc, S=2000, n=n, sigma=SIGMA, seed=SEED)

    with open("results/q3e_identity_check.txt", "w") as f:
        for label, rows in (("S = 200", rows_s200), ("S = 2000", rows_s2000)):
            f.write(f"\n{label}\n")
            f.write(f"{'M':<4}{'sigma^2':<10}{'Bias^2':<10}{'Var':<10}{'Sum':<10}{'Sim.Total':<12}{'Discrepancy':<12}\n")
            for r in rows:
                f.write(f"{r['M']:<4}{r['sigma2']:<10.4f}{r['bias2']:<10.4f}{r['var']:<10.4f}"
                        f"{r['sum']:<10.4f}{r['simulated_total']:<12.4f}{r['discrepancy']:<12.5f}\n")
    print("\n3(e) S=200:")
    for r in rows_s200:
        print(r)
    print("\n3(e) S=2000:")
    for r in rows_s2000:
        print(r)

    bias2_f, var_f, sum_f = part_f(M_list_full, S=200, n=n, seed=SEED,
                                     outpath="figures/q3f_bias_var_vs_degree.png")
    with open("results/q3f_bias_var_vs_degree.txt", "w") as f:
        f.write(f"{'M':<5}{'Bias^2':<12}{'Var':<12}{'Sum':<12}\n")
        for M, b2, v, s in zip(M_list_full, bias2_f, var_f, sum_f):
            f.write(f"{M:<5}{b2:<12.5f}{v:<12.5f}{s:<12.5f}\n")

    print("\nSaved figures/q3b_*, q3f_* and results/q3c_*, q3e_*, q3f_*")

if __name__ == "__main__":
    main()
