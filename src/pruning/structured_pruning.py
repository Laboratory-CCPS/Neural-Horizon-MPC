import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from typing import Literal

from ..torch_utils import count_linear_modules


def ln_structured_fixed_amount(
        model: nn.Module, 
        param_name: Literal['weight', 'bias'], 
        amount: int | float, 
        n: float, 
        dim: int
    ):
    """Prunes *amount* of the lowest Ln norms of a vector in one 
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
            if layer_count == num_layers and 0 == dim: 
                temp_amount = 0.0

            if 1 == layer_count and 1 == dim: 
                temp_amount = 0.0

            prune.ln_structured(module, name=param_name, amount=temp_amount, n=n, dim=dim)
            
    return model



#====================================================================================================
# NEXT OR PERVIOUS PARAMS
#====================================================================================================
class StructuredNextOrPrevMask(prune.BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, prev_mask: torch.Tensor, dim = -1, mask_dim = -1):
        """
        Initializes the mask object with the given parameters.

        Parameters
        ----------
        ``prev_mask`` : torch.Tensor
            The previous mask tensor to be used.
        ``dim`` : int, optional
            The dimension of the new mask. 
            *Default = -1*
        ``mask_dim`` : int, optional
            Specifies the mask dimension type: 0 for previous and 1 for next. 
            *Default = -1*
        """
        super().__init__()
        self.prev_mask = prev_mask
        self.dim = dim
        self.mask_dim = mask_dim

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor):
        """Computes a mask where the weights and biases are pruned 
        because previous weights where pruned. (Removes complete nodes 
        with it's weights and biases) 
        """
        mask = default_mask.clone()

        zero_rows = torch.where(self.prev_mask.count_nonzero(dim=self.mask_dim) == 0)
        if zero_rows[0].shape[0] != 0:
            slc = [slice(None)] * len(t.shape)
            slc[self.dim] = torch.split(*zero_rows, 1)
            mask[slc] = 0
        return mask

    @classmethod
    def apply(cls, module, name, prev_mask, dim, mask_dim):
        return super(StructuredNextOrPrevMask, cls).apply(
            module, name, prev_mask, dim, mask_dim
        )


def structured_next_prev_params(
        module: nn.Module, 
        param_name: Literal['weight', 'bias'], 
        mask: torch.Tensor, 
        dim: int, 
        mask_dim: int
    ):
    """Prunes complete nodes of one *module*, if the *prev_mask* cut the the node of. 
    This pruning technique is structured and just an addition. 

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 
    ``param_name`` : Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned. 
    ``dim`` : int
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node
    ``mask`` : torch.Tensor
        The weight mask of the previous or next module.  
    ``dim`` int
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node
    ``mask_dim`` : int 
        The value is -1 if previous and 1 if next mask is given. 

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned. 
    """
    StructuredNextOrPrevMask.apply(module, param_name, prev_mask=mask, dim=dim, mask_dim=mask_dim)
    return module


#----------------------------------------------------------------------------------------------------
# NEXT PARAMS
def structured_next_params(model: nn.Module):
    """
    Prunes complete nodes of the whole *model*, if the *prev_mask*  of one module 
    cut the node of. This pruning technique is structured and just an addition. 

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 
    ``param_name`` : Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned. 

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned. 
    """
    num_layers = count_linear_modules(model)

    prev_mask = None
    layer_count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            layer_count += 1

            # dont prun next params for input layer
            if prev_mask is not None:
                structured_next_prev_params(module, 'weight', prev_mask, 1, 1)

            prev_mask = list(module.buffers())[0]

            # dont prun bias of output layer
            if layer_count < num_layers:
                structured_next_prev_params(module, 'bias', prev_mask, 0, 1)
    return model


#----------------------------------------------------------------------------------------------------
# PERVIOUS PARAMS
def structured_prev_params(model: nn.Module):
    """
    Prunes complete nodes of the whole *model*, if the *next_mask* of one module 
    cut the node of. This pruning technique is structured and just an addition. 

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned. 
    """
    num_layers = count_linear_modules(model)

    next_weight_masks = [
        list(module.buffers())[0] for module in model.modules() if isinstance(module, nn.Linear)
    ]
    next_mask_idx = 0
    iters = len(next_weight_masks) - 1
    for module in model.modules():
        if isinstance(module, nn.Linear):
            next_mask_idx += 1
            structured_next_prev_params(module, 'weight', next_weight_masks[next_mask_idx], 0, 0)
            structured_next_prev_params(module, 'bias', next_weight_masks[next_mask_idx], 0, 0)
            if iters == next_mask_idx:
                break
            
    return model



#----------------------------------------------------------------------------------------------------
# NODE PRUNING
def nodes_ln_structured(model: nn.Module, param_name: Literal['weight', 'bias'], amount: float, n: float, dim: int):
    """
    Prunes complete nodes of the whole *model* with Ln norm structured 
    pruning. 

    Parameters
    ----------
    ``model`` : nn.Module 
        The model that should be pruned. 
    ``param_name`` : Literal['weight', 'bias']
        A string to determine if the weights or the bias should be pruned. 
    ``amount`` : int | float
        The amount in percent or fixed connections in one module. 
    ``n`` : int | float | inf | -inf | 'fro' | 'nuc'
        The norm to determine the relevance of weights.
    ``dim`` : int
        The dimension of the structured pruning. 
        0 -> all weights that enter one node. 
        1 -> all weights that exit one node

    Returns
    -------
    ``model`` : nn.Module  
        This model is pruned.
    """
    ln_structured_fixed_amount(model, param_name, amount, n, dim)

    if 0 == dim:
        structured_next_params(model)
    elif 1 == dim:
        structured_prev_params(model)
    else:
        raise ValueError(f'Dimension {dim} is not correct and need to be 0 or 1!')
    return model