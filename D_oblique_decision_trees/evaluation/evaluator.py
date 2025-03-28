from .metrics import (
    compute_accuracy, compute_coverage, compute_density, compute_f_score, interpretability_score,
    sparsity, compute_convergence_length, compute_training_time, seed_stability
)


def evaluate_tree(tree, X, y, logs=None, training_time=None, seed_scores=None):
    return {
        "accuracy": compute_accuracy(tree, X, y),
        "coverage": compute_coverage(tree, X, y),
        "density": compute_density(tree, X, y),
        "f_score": compute_f_score(tree, X, y),
        "depth": tree.max_depth,
        "splits": tree.num_splits,
        "leaves": tree.num_leaves,
        "interpretability": interpretability_score(tree),
        "sparsity": sparsity(tree),
        "convergence_length": compute_convergence_length(logs),
        "training_time": training_time,
        "stability_across_seeds": seed_stability(seed_scores) if seed_scores else None
    }
