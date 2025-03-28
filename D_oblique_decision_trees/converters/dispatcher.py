# D_oblique_decision_trees/converters/dispatcher.py

from .hhcart_converter import convert_hhcart
from .randcart_converter import convert_randcart
from .co2_converter import convert_co2
from .oc1_converter import convert_oc1
from .wodt_converter import convert_wodt


def convert_tree(model, model_type):
    if model_type == 'hhcart':
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
