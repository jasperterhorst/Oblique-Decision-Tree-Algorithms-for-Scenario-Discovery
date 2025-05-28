import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_coverage_density_tradeoff(hh, k_range=None, save_path=None):
    if hh.metrics_df is None:
        raise RuntimeError("You must call .build_tree() before plotting.")

    df = hh.metrics_df.copy()
    if "coverage" not in df or "density" not in df:
        raise ValueError("Expected 'coverage' and 'density' columns in metrics_df.")

    if k_range is not None:
        df = df[df["k"].isin(k_range)]

    unique_k = sorted(df["k"].dropna().unique())
    cmap = cm.get_cmap("tab10", len(unique_k))
    color_map = {k: cmap(i) for i, k in enumerate(unique_k)}

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for k in unique_k:
        df_k = df[df["k"] == k].sort_values("depth")
        coverage = df_k["coverage"].values
        density = df_k["density"].values
        depths = df_k["depth"].values
        sizes = 20 + 15 * depths

        ax.plot(coverage, density, '-', color=color_map[k], label=f"k = {k}")
        ax.scatter(coverage, density, s=sizes, color=color_map[k], alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_title("Coverage vs. Density Trade-Off by k")
    ax.grid(True)
    ax.legend(title="k")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"✅ Saved to {save_path}")
    else:
        plt.show()
