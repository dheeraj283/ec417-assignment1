import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from src.poly_fitting import generate_data, true_function, SIGMA
    from src.bias_variance import GRID
except ImportError:
    from poly_fitting import generate_data, true_function, SIGMA
    from bias_variance import GRID

SEED = 42
DEGREE = 9
LAMBDAS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]

def build_poly_features(x, degree):
    x = np.asarray(x).reshape(-1, 1)
    return np.hstack([x ** p for p in range(1, degree + 1)])

def fit_ridge(x_train, y_train, degree, lam):
    Phi_train = build_poly_features(x_train, degree)
    scaler = StandardScaler()
    Phi_train_scaled = scaler.fit_transform(Phi_train)
    cond = np.linalg.cond(Phi_train_scaled)

    model = Ridge(alpha=lam, fit_intercept=True)
    model.fit(Phi_train_scaled, y_train)
    return model, scaler, cond

def predict_ridge(model, scaler, x, degree):
    Phi = build_poly_features(x, degree)
    Phi_scaled = scaler.transform(Phi)
    return model.predict(Phi_scaled)

def part_a(outpath):
    rng = np.random.default_rng(SEED)
    x_train, y_train = generate_data(15, rng)
    x_grid = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(4 * len(LAMBDAS), 4), sharey=True)
    conds = []
    for ax, lam in zip(axes, LAMBDAS):
        model, scaler, cond = fit_ridge(x_train, y_train, DEGREE, lam)
        conds.append(cond)
        y_grid = predict_ridge(model, scaler, x_grid, DEGREE)
        ax.scatter(x_train, y_train, s=20, color="tab:blue", zorder=3)
        ax.plot(x_grid, true_function(x_grid), color="tab:green", lw=2, label=r"$f^\star$")
        ax.plot(x_grid, y_grid, color="tab:red", lw=2, label="Ridge fit")
        ax.set_title(f"$\\lambda$ = {lam:g}")
        ax.set_xlabel("x")
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("y")
    fig.suptitle(f"Degree-9 ridge fits, n=15 (condition numbers logged separately)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    with open("results/q6a_condition_numbers.txt", "w") as f:
        f.write(f"{'lambda':<12}{'cond(Phi_scaled)':<20}\n")
        for lam, c in zip(LAMBDAS, conds):
            f.write(f"{lam:<12g}{c:<20.4e}\n")
    print("Saved figures/q6a + results/q6a_condition_numbers.txt")
    return conds

def part_b(outpath, S=200, n=15):
    bias2_list, var_list = [], []
    for lam in LAMBDAS:
        rng = np.random.default_rng(SEED)
        Fhat = np.zeros((S, len(GRID)))
        for s in range(S):
            x_train, y_train = generate_data(n, rng)
            model, scaler, _ = fit_ridge(x_train, y_train, DEGREE, lam)
            Fhat[s] = predict_ridge(model, scaler, GRID, DEGREE)
        fbar = Fhat.mean(axis=0)
        bias2 = np.mean((fbar - true_function(GRID)) ** 2)
        var = np.mean(Fhat.var(axis=0))
        bias2_list.append(bias2)
        var_list.append(var)

    log_lams = np.log10(LAMBDAS)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(log_lams, bias2_list, "o-", label=r"Bias$^2$")
    ax.plot(log_lams, var_list, "o-", label="Variance")
    ax.set_xlabel(r"$\log_{10} \lambda$")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.set_title("Degree-9 ridge: Bias$^2$/Variance vs. $\\log_{10}\\lambda$ (n=15)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    with open("results/q6b_bias_var_vs_lambda.txt", "w") as f:
        f.write(f"{'lambda':<12}{'Bias^2':<14}{'Var':<14}\n")
        for lam, b2, v in zip(LAMBDAS, bias2_list, var_list):
            f.write(f"{lam:<12g}{b2:<14.5f}{v:<14.5f}\n")
    print("Saved figures/q6b + results/q6b_bias_var_vs_lambda.txt")
    return bias2_list, var_list

def main():
    conds = part_a("figures/q6a_ridge_fits.png")
    print("Condition numbers by lambda:", dict(zip(LAMBDAS, conds)))
    bias2_list, var_list = part_b("figures/q6b_bias_var_vs_lambda.png")
    print("Bias^2 by lambda:", dict(zip(LAMBDAS, bias2_list)))
    print("Var by lambda:", dict(zip(LAMBDAS, var_list)))

if __name__ == "__main__":
    main()
