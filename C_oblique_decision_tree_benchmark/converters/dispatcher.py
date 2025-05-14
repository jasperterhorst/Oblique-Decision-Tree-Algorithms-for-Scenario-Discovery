"""
Dispatcher module for model converters.

Provides a single interface to convert different model types into the standardized DecisionTree.
"""

from C_oblique_decision_tree_benchmark.converters.hhcart_converter import convert_hhcart
from C_oblique_decision_tree_benchmark.converters.randcart_converter import convert_randcart
from C_oblique_decision_tree_benchmark.converters.co2_converter import convert_co2
from C_oblique_decision_tree_benchmark.converters.oc1_converter import convert_oc1
from C_oblique_decision_tree_benchmark.converters.wodt_converter import convert_wodt
from C_oblique_decision_tree_benchmark.converters.ridgecart_converter import convert_ridge_cart
from C_oblique_decision_tree_benchmark.converters.cart_converter import convert_cart


def convert_tree(model, model_type):
    """
    Convert a given model into a standardized DecisionTree based on the model type.

    Parameters:
        model: The trained model instance.
        model_type (str): The type identifier for the model. Accepted values:
                          'hhcart_a', 'hhcart_d', 'randcart', 'co2', 'oc1',
                          'wodt', 'ridge_cart', 'cart'.

    Returns:
        DecisionTree: The converted decision tree.

    Raises:
        ValueError: If the provided model_type is unsupported.
    """
    if model_type in {'hhcart_a', 'hhcart_d'}:
        return convert_hhcart(model)
    elif model_type == 'randcart':
        return convert_randcart(model)
    elif model_type == 'co2':
        return convert_co2(model)
    elif model_type == 'oc1':
        return convert_oc1(model)
    elif model_type == 'wodt':
        return convert_wodt(model)
    elif model_type == 'ridge_cart':
        return convert_ridge_cart(model)
    elif model_type == 'cart':
        return convert_cart(model)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
