import torch.nn as nn
import torch.nn.utils.prune as prune

from typing import Literal

from ..torch_utils import count_linear_modules
from .structured_pruning import structured_prev_params, structured_next_params



def global_unstructured_fixed_amount(
        model: nn.Module, 
        param_name: Literal['weight', 'bias'], 
        amount: int | float
    ):
    """
    Prunes ``amount`` of the lowest L1 norms in the whole ``model`` of 
    bias or weights. This pruning technique is global and unstructured.  

    Parameters
    ----------
    ``model`` : nn.Module
        This is the model that should be pruned. 
    ``param_name`` :  Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned.
    ``amount`` : int | float
        The amount in percent or fixed connections in the model. 

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned.
    """
    parameters_to_prune = [
        (module, param_name) 
        for module in filter(lambda m: type(m) == nn.Linear, model.modules())
    ]
    prune.global_unstructured(
        parameters_to_prune, 
        pruning_method=prune.L1Unstructured, 
        amount=amount
    )
    return model



#----------------------------------------------------------------------------------------------------
# NODE PRUNING
def nodes_global_unstructured(model: nn.Module, param_name: Literal['weight', 'bias'], amount: float):
    """
    Prunes complete nodes of the *model* if the previous or next weights are all pruned 
    from the global unstructured pruning. 

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 
    ``param_name`` : Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned. 
    ``amount`` : int | float
        The amount in percent or fixed connections in one module. 

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned.
    """
    global_unstructured_fixed_amount(model, param_name, amount)

    structured_prev_params(model)
    structured_next_params(model)
    return model