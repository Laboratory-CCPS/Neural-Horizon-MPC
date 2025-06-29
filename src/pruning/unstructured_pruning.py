import torch.nn as nn
import torch.nn.utils.prune as prune

from typing import Literal

from ..torch_utils import count_linear_modules
from .structured_pruning import structured_prev_params, structured_next_params



def l1_unstructured_fixed_amount(
        model: nn.Module, 
        param_name: Literal['weight', 'bias'], 
        amount: int | float,
        prune_last_layer: bool = True,
    ):
    """Prunes *amount* of the lowest L1 norms of a vector in one 
    weight matrix or bias vector. It is used for the whole *model*, 
    but every module at once, with the same 'amount'.  

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 
    ``param_name`` : Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned. 
    ``amount`` : int | float
        The amount in percent or fixed connections in one module. 
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc'
        The the norm. 
    ``dim`` : int
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned. 
    """
    num_layers = count_linear_modules(model)

    layer_count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            layer_count += 1
            temp_amount = amount

            # find output layer and set amount to 0
            if not prune_last_layer and layer_count == num_layers: 
                temp_amount = 0.0

            prune.l1_unstructured(module, name=param_name, amount=temp_amount)
            
    return model




#----------------------------------------------------------------------------------------------------
# NODE PRUNING
def nodes_l1_unstructured(model: nn.Module, param_name: Literal['weight', 'bias'], amount: float):
    """
    Prunes complete nodes of the *model* if the previous or next weights are all pruned 
    from the L1 norm unstructured pruning. 

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
    l1_unstructured_fixed_amount(model, param_name, amount)

    structured_prev_params(model)
    structured_next_params(model)
    return model