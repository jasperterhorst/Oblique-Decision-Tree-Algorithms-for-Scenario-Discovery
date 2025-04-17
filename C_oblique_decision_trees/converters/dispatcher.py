"""
Dispatcher module for model converters.

Provides a single interface to convert different model types into the standardized DecisionTree.
"""

from C_oblique_decision_trees.converters.hhcart_converter import convert_hhcart
from C_oblique_decision_trees.converters.randcart_converter import convert_randcart
from C_oblique_decision_trees.converters.co2_converter import convert_co2
from C_oblique_decision_trees.converters.oc1_converter import convert_oc1
from C_oblique_decision_trees.converters.wodt_converter import convert_wodt


def convert_tree(model, model_type):
    """
    Convert a given model into a standardized DecisionTree based on the model type.

    Parameters:
        model: The trained model instance.
        model_type (str): The type identifier for the model. Accepted values:
                          'hhcart_a', 'hhcart_d', 'randcart', 'co2', 'oc1', 'wodt', 'cart_lc'.

    Returns:
        DecisionTree: The converted decision tree.

    Raises:
        ValueError: If the provided model_type is unsupported.
    """
    if model_type == 'hhcart_a':
        return convert_hhcart(model)
    elif model_type == 'hhcart_d':
        return convert_hhcart(model)
    elif model_type == 'randcart':
        return convert_randcart(model)
    elif model_type == 'co2':
        return convert_co2(model)
    elif model_type == 'oc1':
        return convert_oc1(model)
    elif model_type == 'wodt':
        return convert_wodt(model)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
