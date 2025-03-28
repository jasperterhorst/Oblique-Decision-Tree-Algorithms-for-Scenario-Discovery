import numpy as np
import math


def predict_subtree(node, x):
    """
    Recursively predict the label for x using the tree.
    For an internal node, the decision is made via the weighted sum computed from
    the features at the indices given by node.split.attrIDs.
    If a leaf is reached, return node.value if available, or else use the mode of
    node.class_distribution.
    """
    if node is None:
        raise ValueError("Encountered a None node during prediction.")
    if node.is_leaf:
        if hasattr(node, "value"):
            return node.value
        elif hasattr(node, "class_distribution"):
            # Return the label with highest frequency.
            return np.argmax(node.class_distribution)
        else:
            raise AttributeError("Leaf node missing 'value' or 'class_distribution'.")
    else:
        d = x.shape[0]
        w = np.zeros(d)
        for idx, coef in zip(node.split.attrIDs, node.split.paras):
            if idx < d:
                w[idx] = coef
        threshold = node.split.threshold
        if np.dot(w, x) < threshold:
            return predict_subtree(node.LChild, x)
        else:
            return predict_subtree(node.RChild, x)


def get_node_params(node, d):
    """
    Returns a parameter vector of length (d+1): first d are weights (according to attrIDs) and the last is threshold.
    """
    params = np.zeros(d + 1)
    if hasattr(node.split, "attrIDs") and hasattr(node.split, "paras") and hasattr(node.split, "threshold"):
        for idx, coef in zip(node.split.attrIDs, node.split.paras):
            if idx < d:
                params[idx] = coef
        params[d] = node.split.threshold
    else:
        raise ValueError("The node.split object does not have the required attributes.")
    return params


def update_node_params(node, params):
    """
    Updates the node's split parameters from the parameter vector (length d+1).
    Assumes the same attrIDs remain.
    """
    d = len(params) - 1
    new_paras = []
    for idx in node.split.attrIDs:
        if idx < d:
            new_paras.append(params[idx])
        else:
            new_paras.append(0)
    node.split.paras = new_paras
    node.split.threshold = params[d]


def is_pure(labels):
    """Return True if all labels are equal."""
    return np.all(labels == labels[0])


def make_leaf(labels):
    """
    Creates a leaf node. Here we simply set the value to the mode (most frequent label).
    You may also set a class_distribution attribute.
    """
    # For simplicity, assume labels are integers.
    counts = np.bincount(labels)
    mode = np.argmax(counts)
    leaf = type('LeafNode', (), {})()  # create a simple empty object
    leaf.is_leaf = True
    leaf.value = mode
    # Optionally, you could also add a class_distribution:
    leaf.class_distribution = counts / np.sum(counts)
    return leaf


def tao_optimize_node(node, X, y, indices, learning_rate=0.01, num_iter=50, min_samples=0,
                      purity_threshold=0, lambda_reg=0.0):
    """
    Recursively optimize the node parameters using only the samples that reach this node.
    In addition, if one branch becomes dead or pure, prune that branch.
    """
    if node is None:
        return
    if node.is_leaf:
        return

    # print(f"\n[Node Depth {getattr(node, 'depth', '?')}] Received {len(indices)} samples.")

    d = X.shape[1]
    # Gather diagnostic info and "care" points.
    care_indices = []
    care_targets = []  # 0 if left should be chosen, 1 if right
    count_both_correct = 0
    count_both_wrong = 0
    count_left_correct = 0
    count_right_correct = 0

    for i in indices:
        x_i = X[i]
        true_label = y[i]
        pred_left = predict_subtree(node.LChild, x_i)
        pred_right = predict_subtree(node.RChild, x_i)
        loss_left = 0 if pred_left == true_label else 1
        loss_right = 0 if pred_right == true_label else 1

        if loss_left == 0 and loss_right == 0:
            count_both_correct += 1
        elif loss_left == 1 and loss_right == 1:
            count_both_wrong += 1
        elif loss_left == 0 and loss_right == 1:
            care_indices.append(i)
            care_targets.append(0)
            count_left_correct += 1
        elif loss_left == 1 and loss_right == 0:
            care_indices.append(i)
            care_targets.append(1)
            count_right_correct += 1

    # print("Diagnostics:")
    # print(f"  Total samples: {len(indices)}")
    # print(f"  Both children correct: {count_both_correct}")
    # print(f"  Both children wrong: {count_both_wrong}")
    # print(f"  Left correct only: {count_left_correct}")
    # print(f"  Right correct only: {count_right_correct}")
    # print(f"  Care points identified: {len(care_indices)}")

    if len(care_indices) > 0:
        X_care = X[care_indices]
        y_care = np.array(care_targets)
        n_care = X_care.shape[0]
        X_aug = np.hstack([X_care, np.ones((n_care, 1))])
        params = get_node_params(node, d)

        def sigmoid(z):
            z = np.clip(z, -700, 700)
            return 1.0 / (1 + np.exp(-z))

        for it in range(num_iter):
            scores = X_aug.dot(params)
            preds = sigmoid(scores)
            loss = -np.mean(y_care * np.log(preds + 1e-8) + (1 - y_care) * np.log(1 - preds + 1e-8))
            grad = X_aug.T.dot(preds - y_care) / n_care
            # Add L1 penalty if lambda_reg > 0
            if lambda_reg > 0:
                grad += lambda_reg * np.sign(params)
            params -= learning_rate * grad
    #         if it % 10 == 0:
    #             print(f"    Iter {it}: surrogate loss = {loss:.4f}")
    #     print(f"  Updated node parameters: {params}")
    #     update_node_params(node, params)
    # else:
    #     print("  No care points; skipping parameter update for this node.")

    # Partition the samples based on updated decision.
    new_params = get_node_params(node, d)
    left_indices = []
    right_indices = []
    for i in indices:
        x_i = X[i]
        if np.dot(new_params[:d], x_i) < new_params[d]:
            left_indices.append(i)
        else:
            right_indices.append(i)
    # print(f"  Partitioning: {len(left_indices)} samples to left; {len(right_indices)} to right.")

    # More aggressive pruning:
    # Dead branch: if a branch gets fewer than min_samples, prune.
    if len(left_indices) < min_samples or len(right_indices) < min_samples:
        # print("  Dead branch detected (fewer than minimum samples); pruning current node.")
        pruned_leaf = make_leaf(y[indices])
        node.is_leaf = True
        node.value = pruned_leaf.value
        if hasattr(pruned_leaf, "class_distribution"):
            node.class_distribution = pruned_leaf.class_distribution
        node.LChild = None
        node.RChild = None
        node.split = None
        return

    # Purity: if a branch is sufficiently pure (e.g., impurity below threshold), prune that branch.
    if len(left_indices) > 0 and (np.mean(y[left_indices] != np.bincount(y[left_indices]).argmax()) < purity_threshold):
        # print("  Left branch is sufficiently pure; pruning left branch.")
        node.LChild = make_leaf(y[left_indices])
    if len(right_indices) > 0 and (
            np.mean(y[right_indices] != np.bincount(y[right_indices]).argmax()) < purity_threshold):
        # print("  Right branch is sufficiently pure; pruning right branch.")
        node.RChild = make_leaf(y[right_indices])

    # Recurse on children.
    tao_optimize_node(node.LChild, X, y, left_indices, learning_rate, num_iter, min_samples, purity_threshold,
                      lambda_reg)
    tao_optimize_node(node.RChild, X, y, right_indices, learning_rate, num_iter, min_samples, purity_threshold,
                      lambda_reg)

def tao_optimize(tree, X, y, num_passes=3, learning_rate=0.01, num_iter=50,
                 min_samples=0, purity_threshold=0, lambda_reg=0.0):
    """
    Runs TAO over the entire tree for a number of passes.
    """
    n_samples = X.shape[0]
    all_indices = list(range(n_samples))
    for p in range(num_passes):
        # print(f"\n========== TAO Pass {p + 1}/{num_passes} ==========")
        tao_optimize_node(tree.root_node, X, y, all_indices, learning_rate, num_iter,
                          min_samples, purity_threshold, lambda_reg)
    # print("\nTAO optimization complete.")
    return tree
