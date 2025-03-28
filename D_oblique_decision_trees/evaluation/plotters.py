import matplotlib.pyplot as plt


def plot_accuracy_vs_depth(results_df):
    """
    Plot Accuracy vs. Depth for each model.
    """
    plt.figure(figsize=(8, 6))
    for model in results_df["algorithm"].unique():
        model_data = results_df[results_df["algorithm"] == model]
        plt.plot(model_data["depth"], model_data["accuracy"], label=model)

    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Tree Depth')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def plot_sparsity_vs_depth(results_df):
    """
    Plot Sparsity vs. Depth for each model.
    """
    plt.figure(figsize=(8, 6))
    for model in results_df["algorithm"].unique():
        model_data = results_df[results_df["algorithm"] == model]
        plt.plot(model_data["depth"], model_data["sparsity"], label=model)

    plt.xlabel('Tree Depth')
    plt.ylabel('Sparsity')
    plt.title('Sparsity vs. Tree Depth')
    plt.legend()
    plt.show()


def plot_convergence_length_vs_depth(results_df):
    """
    Plot Convergence Length vs. Depth for each model.
    """
    plt.figure(figsize=(8, 6))
    for model in results_df["algorithm"].unique():
        model_data = results_df[results_df["algorithm"] == model]
        plt.plot(model_data["depth"], model_data["convergence_length"], label=model)

    plt.xlabel('Tree Depth')
    plt.ylabel('Convergence Length')
    plt.title('Convergence Length vs. Tree Depth')
    plt.legend()
    plt.show()
