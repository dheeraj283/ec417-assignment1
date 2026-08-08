import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
_FALLBACK_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data_cal_housing.csv")

def load_df():
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        return data.frame.copy()
    except Exception as e:
        if os.path.exists(_FALLBACK_CSV):
            print(f"[load_df] sklearn download failed ({e}); using cached "
                  f"local CSV at {_FALLBACK_CSV} instead.")
            return pd.read_csv(_FALLBACK_CSV)
        raise

def part_a(df):
    with open("results/q4a_summary.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Dtypes:\n")
        f.write(df.dtypes.to_string())
        f.write("\n\nMissing values per column:\n")
        f.write(df.isna().sum().to_string())
        f.write("\n\nSummary statistics:\n")
        f.write(df.describe().to_string())
    print("Saved results/q4a_summary.txt")

def part_b(df, outpath):
    target = df["MedHouseVal"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(target, bins=50, color="tab:blue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Median house value ($100,000s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of target: Median House Value")
    ax.axvline(target.max(), color="red", ls="--", label=f"max = {target.max():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    n_at_cap = (target == target.max()).sum()
    with open("results/q4b_topcoding.txt", "w") as f:
        f.write(f"Target max value: {target.max()}\n")
        f.write(f"Number of districts exactly at the max value: {n_at_cap} "
                f"({100 * n_at_cap / len(target):.2f}% of data)\n")
        f.write(
            "Phenomenon: right-censoring / top-coding. The target was capped at "
            "$500,000 (5.00 in units of $100,000) by the data provider (US Census), "
            "so any district whose true median value exceeds the cap is recorded "
            "at exactly the cap value. A regression model trained on this data "
            "systematically underestimates the value of expensive districts: it "
            "cannot learn to predict above the cap, and it is trained to match a "
            "target value that is wrong (too low) for all districts above the "
            "true $500k threshold, biasing predictions downward for high-value areas.\n"
        )
    print(f"Saved figures/q4b + results/q4b_topcoding.txt (n at cap = {n_at_cap})")

def part_c(df, outpath):
    """4(c): histograms of all 8 predictors."""
    predictors = [c for c in df.columns if c != "MedHouseVal"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), predictors):
        ax.hist(df[col], bins=50, color="tab:orange", edgecolor="black", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    fig.suptitle("Histograms of all 8 predictors")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    skews = df[predictors].skew().sort_values(ascending=False)
    maxes = df[predictors].max()
    with open("results/q4c_skew.txt", "w") as f:
        f.write("Skewness of each predictor (descending):\n")
        f.write(skews.to_string())
        f.write("\n\nMax value of each predictor:\n")
        f.write(maxes.to_string())
        f.write(
            "\n\nCandidates for implausible extreme values (inspect these against "
            "the histograms in figures/q4c_predictor_histograms.png):\n"
            "  - AveRooms / AveBedrms: max values far exceed any plausible average "
            "rooms-per-household for a census block (likely a block with very few "
            "households, making the average highly sensitive to outliers / data errors).\n"
            "  - AveOccup: max value is implausibly large for average household "
            "occupancy (likely the same small-denominator issue, or group-quarters "
            "populations such as institutions).\n"
            "  - Population: some blocks show very large or very small populations "
            "relative to the typical block group size.\n"
            "Proposed treatment: do not simply delete these rows. Instead, (i) cap/"
            "winsorize at a high percentile (e.g. 99th) to limit leverage, since "
            "removal discards real (if unusual) block groups and can bias the "
            "remaining sample; (ii) inspect the denominator (households) for these "
            "rows -- if households is very small, the ratio features are numerically "
            "unstable and a robust transform (e.g. log) or a minimum-household filter "
            "is more principled than outright deletion.\n"
        )
    print("Saved figures/q4c + results/q4c_skew.txt")

def part_d(df, outpath):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation matrix heatmap")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    target_corr = corr["MedHouseVal"].drop("MedHouseVal").abs().sort_values(ascending=False)
    corr_no_diag = corr.copy(deep=True)
    vals = corr_no_diag.values.copy()
    np.fill_diagonal(vals, 0)
    corr_no_diag = pd.DataFrame(vals, index=corr_no_diag.index, columns=corr_no_diag.columns)
    corr_no_diag = corr_no_diag.drop(columns=["MedHouseVal"], errors="ignore")
    corr_no_diag = corr_no_diag.drop(index=["MedHouseVal"], errors="ignore")
    max_pair_val = corr_no_diag.abs().values.max()
    idx = np.unravel_index(np.argmax(corr_no_diag.abs().values), corr_no_diag.shape)
    pair = (corr_no_diag.index[idx[0]], corr_no_diag.columns[idx[1]])

    with open("results/q4d_correlations.txt", "w") as f:
        f.write("Correlation with target (|r|, descending):\n")
        f.write(target_corr.to_string())
        f.write(f"\n\nStrongest predictor-predictor correlation (excluding target): "
                f"{pair[0]} vs {pair[1]} = {max_pair_val:.4f}\n")
        f.write(
            "This matters because strongly correlated predictors (multicollinearity) "
            "make individual regression coefficients unstable and hard to interpret "
            "(the model can trade weight between the two correlated features "
            "without changing predictions much), even though it may not hurt raw "
            "predictive accuracy much.\n"
        )
    print("Saved figures/q4d + results/q4d_correlations.txt")

def part_e(df, outpath):
    """4(e): scatter of Longitude vs Latitude colored by target."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                     cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Median house value ($100,000s)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic distribution of median house value")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    with open("results/q4e_geography.txt", "w") as f:
        f.write(
            "Two dense high-value clusters are visible: one around the San "
            "Francisco Bay Area (~ -122.4 lon, ~37.7-37.9 lat) and one around the "
            "Los Angeles / coastal Southern California area (~ -118.2 to -117.9 "
            "lon, ~33.7-34.1 lat).\n\n"
            "Neighboring census blocks share unobserved geographic covariates "
            "(school quality, distance to the coast, local zoning, employer "
            "proximity, crime, amenities) that are not in the eight predictors. "
            "This means nearby rows are correlated rather than independent draws, "
            "violating the i.i.d. assumption. Consequence for a random train/test "
            "split: because nearby blocks end up on both sides of the split, "
            "performance metrics are optimistic (a form of leakage) -- the model "
            "can partly 'memorize' the local neighborhood pattern from a training "
            "block and exploit it on a geographically adjacent test block, so a "
            "random split overstates how well the model will generalize to a "
            "genuinely new, geographically distinct area. A spatial (e.g. block-"
            "group or region-based) split would give a more honest estimate.\n"
        )
    print("Saved figures/q4e + results/q4e_geography.txt")

def part_f(df):
    """4(f): data leakage demonstration -- standardize before vs after split."""
    predictors = [c for c in df.columns if c != "MedHouseVal"]
    X = df[predictors].values
    y = df["MedHouseVal"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # Method (i): fit scaler on train only
    mu_train = X_train.mean(axis=0)
    sigma_train = X_train.std(axis=0)
    X_test_scaled_i = (X_test - mu_train) / sigma_train

    # Method (ii): fit scaler on entire dataset before splitting
    mu_all = X.mean(axis=0)
    sigma_all = X.std(axis=0)
    _, X_test_full_scaled_source, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    X_test_scaled_ii = (X_test_full_scaled_source - mu_all) / sigma_all

    mean_i = X_test_scaled_i.mean(axis=0)
    mean_ii = X_test_scaled_ii.mean(axis=0)
    diff = mean_i - mean_ii

    with open("results/q4f_leakage.txt", "w") as f:
        f.write(f"{'Feature':<12}{'TestMean(i) train-only':<26}{'TestMean(ii) full-data':<26}{'Diff':<12}\n")
        for name, mi, mii, d in zip(predictors, mean_i, mean_ii, diff):
            f.write(f"{name:<12}{mi:<26.5f}{mii:<26.5f}{d:<12.5f}\n")
        f.write(
            "\nMethod (i) -- fitting the scaler on the training split only -- is the "
            "only correct procedure because the test set must remain completely "
            "unseen during preprocessing, exactly as it would be unseen at "
            "deployment time. Method (ii) leaks information from the test set "
            "(its values contribute to the mean/std used to scale it), which can "
            "make validation performance look better than true generalization "
            "performance. Here the difference is small because the dataset is "
            "large (n=20,640) and randomly split, so train and full-data statistics "
            "are numerically close.\n\n"
            "A realistic situation where the difference would NOT be small: a much "
            "smaller dataset, a non-random or time-based split (e.g. training on "
            "earlier years and testing on later years, where the distribution "
            "shifts over time), or a split with severe class/group imbalance "
            "(e.g. splitting by geographic region so train and test come from "
            "different underlying populations). In any of these cases the "
            "train-only statistics can differ substantially from the full-data "
            "statistics, and using the full-data statistics would leak a "
            "meaningfully biased signal into the test evaluation.\n"
        )
    print("Saved results/q4f_leakage.txt")

def main():
    df = load_df()
    part_a(df)
    part_b(df, "figures/q4b_target_histogram.png")
    part_c(df, "figures/q4c_predictor_histograms.png")
    part_d(df, "figures/q4d_correlation_heatmap.png")
    part_e(df, "figures/q4e_geo_scatter.png")
    part_f(df)
    print("\nProblem 4 complete.")

if __name__ == "__main__":
    main()
