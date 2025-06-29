import pandas as pd
import os
import pickle as pkl

from itertools import product
from typing import Optional
from collections import defaultdict
from functools import reduce

from .mpc_dataclass import AMPC_data, MPC_data 
from .parameters import NH_AMPC_Param



# ------------------------------------------------------------------------------------------------------
## FEATURES AND LABELS
# ------------------------------------------------------------------------------------------------------
def get_features_and_labels(nh_mpc_params: NH_AMPC_Param) -> tuple[list[str], list[str]]:
    """
    Calculates a list of all needed feature and label names based of 
    of the given neural horizon parameter setup.

    Parameters
    ----------
    ``nh_mpc_params`` : NH_AMPC_Param
        Object containing parameters for neural horizon model configuration.

    Returns
    -------
    ``features`` : list[str]
        List of feature names generated based on the parameter setup.
    ``labels`` : list[str]
        List of label names generated based on the parameter setup.
    """
    if 'begin' == nh_mpc_params.DS_begin:
        features = [f'{x}_p{0}' for x in nh_mpc_params.xlabel]
        labels = [f'{x}_p{i+1}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.xlabel]
    elif 'fixed' == nh_mpc_params.DS_begin:
        features = [f'{x}_p{nh_mpc_params.DS_feature}' for x in nh_mpc_params.xlabel]
        labels = [f'{x}_p{i+1+nh_mpc_params.DS_feature}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.xlabel]
    elif '' == nh_mpc_params.DS_begin:
        features = [f'{x}_p{nh_mpc_params.N_MPC}' for x in nh_mpc_params.xlabel]
        labels = [f'{x}_p{i+1+nh_mpc_params.N_MPC}' for i in range(nh_mpc_params.N_NN) for x in nh_mpc_params.xlabel]
    else:
        raise ValueError('Unrecognized value for NH_AMPC_Param.DS_begin! Need to be "begin", "fixed" or "".')
    return features, labels


# ------------------------------------------------------------------------------------------------------
## HIDDEN SIZES
# ------------------------------------------------------------------------------------------------------
def get_hidden_neurons(num: int, start_sizes: list[int] = None):
    """
    Generates a list of hidden neuron sizes for a neural network.

    Parameters
    ----------
    ``num`` : int
        Number of hidden layers to generate.
    ``start_sizes`` : list[int], optional
        Initial sizes of the first hidden layers.
        Default = None.

    Returns
    -------
    ``n_hidden`` : list[int]
        List of hidden neuron sizes for each layer.
    """
    if num <= 0:
        raise ValueError('\"num\" need to be positive!')
    
    n_hidden = [12, 16] if start_sizes is None else start_sizes

    if num <= 2:
        return n_hidden[:num-1]
    
    for i in range(2, num):
        next = n_hidden[i-2] * 2
        n_hidden.append(next)
    return n_hidden


# ------------------------------------------------------------------------------------------------------
## ALL ACADOS OPTIONS
# ------------------------------------------------------------------------------------------------------
def get_all_acados_options(base_cname = 'AMPC_option', print_all_options = False):
    """
    Computes a dictionary with all relevant solver option combinations. 

    Parameters
    ----------
    ``base_cname`` : str
        The base controller name for the keys of the output dictionary.
    ``print_all_options`` : bool, optional
        Whether to print all the option combinations.
        Default = False.

    Returns
    -------
    ``all_acados_options`` : dict
        A dictionary where keys are the base name with an index and values are unique combinations of acados options.
    """
    acados_options = dict(
        nlp_solver_type = ['SQP_RTI', 'SQP'],
        qp_solver = ['FULL_CONDENSING_QPOASES', 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_HPIPM', 
                    'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP'],
        integrator_type = ['ERK', 'IRK', 'DISCRETE', 'LIFTED_IRK'], # 'GNSF'
        hessian_approx = ['GAUSS_NEWTON', 'EXACT'],
        hpipm_mode = ['BALANCE', 'SPEED_ABS', 'SPEED', 'ROBUST'],
        regularize_method = ['NO_REGULARIZE', 'MIRROR', 'PROJECT', 'PROJECT_REDUC_HESS', 'CONVEXIFY'],
        # collocation_type = ['GAUSS_RADAU_IIA', 'GAUSS_LEGENDRE'],
        # globalization = ['FIXED_STEP', 'MERIT_BACKTRACKING'],
        as_rti_iter = [1, 5],
        as_rti_level = [3]
    )

    # Generate all combinations, exclude hpipm_mode
    option_keys = [key for key in acados_options if key not in ['hpipm_mode', 'regularize_method', 'as_rti_iter', 'as_rti_level']]
    option_combinations = list(product(*[acados_options[key] for key in option_keys]))

    # Create a list of dictionaries where each dictionary represents a combination of options
    option_combinations = [dict(zip(option_keys, values)) for values in option_combinations]

    def generate_combinations(base_combination: list[dict], options: dict, option_key: list[str]):
        """
        Generates a list of combinations that have a base combination, 
        which is combined with other options.

        Parameters
        ----------
        ``base_combination`` : list[dict]
            A list of dicts that define an acados option combination.
        ``options``: dict
            A dict with every relevant solver option  from acados.
        ``option_key`` : list[str]
            An option key, which is used as a key for ``options``, to get the right ones.

        Returns
        ------
        ``multi_acados_options`` : list[dict]
            A list combinations based on one combination, combined with other options.
        """
        combinations = list(product(base_combination, options[option_key]))
        return [dict(**values[0], **{option_key: values[1]}) for values in combinations]

    # Add hpipm_mode for HPIPM qp_solver combinations and regularize_methods
    _option_combinations = []
    for base_combination in option_combinations:
        qp_solver = base_combination['qp_solver']
        hessian_approx = base_combination['hessian_approx']
        nlp_solver_type = base_combination['nlp_solver_type']
        integrator_type = base_combination['integrator_type']

        # skip lifted_irk and exact hessian -> does not work
        if 'LIFTED_IRK' == integrator_type and hessian_approx == 'EXACT':
            continue

        base_combs = [base_combination]

        # set RTI to AS-RTI-D since only this works good
        if 'SQP_RTI' == nlp_solver_type:
            base_combs = generate_combinations(base_combs, acados_options, 'as_rti_iter')
            base_combs = generate_combinations(base_combs, acados_options, 'as_rti_level')

        if 'HPIPM' in qp_solver:
            base_combs = generate_combinations(base_combs, acados_options, 'hpipm_mode')

        if 'EXACT' == hessian_approx:
            base_combs = generate_combinations(base_combs, acados_options, 'regularize_method')

        _option_combinations.extend(base_combs)


    option_combinations = _option_combinations

    # Print the option combinations
    if print_all_options:
        for base_combination in option_combinations:
            print(base_combination)

    return {f'{base_cname}_{idx}':combination for idx, combination in enumerate(option_combinations)}



# ------------------------------------------------------------------------------------------------------
## SAVEING LOADING
# ------------------------------------------------------------------------------------------------------
def load_results(file_path: str) -> list[AMPC_data | MPC_data] | None:
    """
    Load pickle results from a ``file_path``.

    Parameters
    ----------
    ``file_path`` (str):
        The path to the file containing the results to be loaded. 

    Returns
    -------
    ``loaded_file`` (object, None):
        Returns the loaded object from the pickle file if the file exists.
        If the file does not exist, prints an error message and returns None.
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as handle:
            return pkl.load(handle)
    else:
        raise ValueError(f'No such file path existent: {file_path}')
    

def save_results(file_path: os.PathLike, results, always_overwrite=False):
    """
    Saves the results to a pickle file in  ``file_path``.

    Parameters
    ----------
    ``file_path`` : PathLike
        The path, where the results should be saved. 
    ``results`` : Any
        Any serializable object that should be saved.

    Returns
    -------
    ``is_saved`` : bool
        A boot defining if the file is sucessfully saved.
    """
    parant_dir = os.path.dirname(file_path)
    if not os.path.exists(parant_dir):
        os.makedirs(parant_dir)

    if os.path.exists(file_path) and not always_overwrite:
        inp = input('Overwrite existing file? [y/n]')
        
        if 'Y' != inp.capitalize():
            return None

    with open(file_path, 'wb') as handle:
        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return os.path.exists(file_path)



# ------------------------------------------------------------------------------------------------------
## FILTERS
# ------------------------------------------------------------------------------------------------------
def get_masked_df(
        df: pd.DataFrame, 
        labels: tuple[str, str], 
        and_filter_dict: dict, 
        or_filter_dict: dict
    ):
    """
    Filters a DataFrame based on the specified 'and' and 'or' conditions.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        The DataFrame to be filtered.
    ``labels`` : tuple[str, str]
        A tuple of column names used for filtering.
    ``and_filter_dict`` : dict
        Dictionary of 'and' conditions where keys are column names and 
         values are single values or lists of values to match.
    ``or_filter_dict`` : dict
        Dictionary of 'or' conditions where keys are column names and 
         values are single values or lists of values to match.

    Returns
    -------
    ``filtered_df`` : pd.DataFrame
        The DataFrame after applying the 'and' and 'or' filters.
    """
    if not or_filter_dict and not and_filter_dict:
        return df
    
    or_mask = None
    if or_filter_dict and not set(or_filter_dict.keys()).issubset(labels):
        or_masks = []
        for key, val in or_filter_dict.items():
            if key not in labels and not isinstance(val, list):
                or_masks.append((df[key] == val))
            elif key not in labels and isinstance(val, list):
                or_masks.extend([(df[key] == v) for v in val])
        
        if or_masks:
            or_mask = reduce(lambda x, y: x | y, or_masks)
    
    and_mask = None
    if and_filter_dict and not set(and_filter_dict.keys()).issubset(labels):
        and_masks = []
        for key, val in and_filter_dict.items():
            if key not in labels and not isinstance(val, list):
                and_masks.append((df[key] == val))
            elif key not in labels and isinstance(val, list):
                and_masks.extend([(df[key] == v) for v in val])

        if and_masks:
            and_mask = reduce(lambda x, y: x & y, and_masks)

    if or_mask is not None and and_mask is not None:
        return df[and_mask & or_mask]
    elif or_mask is not None:
        return df[or_mask]
    elif and_mask is not None:
        return df[and_mask]
    else:
        return df


def get_masked_results(
        results: list[AMPC_data | MPC_data], 
        and_filter_dict: Optional[dict[str, list[object] | object]] = None, 
        or_filter_dict: Optional[dict[str, list[object] | object]] = None, 
        ignore_keys: Optional[tuple[str, ...]] = None
    ) -> list[AMPC_data | MPC_data]:
    """
    Filters the list of results by specified parameters 
    in the and- and or-filter dict.

    Parameters
    ----------
    ``results`` : list[AMPC_data | MPC_data]
        A list of the results that should be filtered.
    ``and_filter_dict`` : Optional[dict[str, list[object] | object]]
        A dictionary with all the keyword arguments of the parameters that
         should be filtered as a set of only the results that match all arguments.
          e.g. : {'N_NN' : 22, 'N_hidden' : 24} 
    ``or_filter_dict`` : Optional[dict[str, list[object] | object]]
        A dictionary with all the keyword arguments of the parameters that 
         should be filtered as a set of all results that match at least one argument.
          e.g. : {'N_NN' : 22, 'N_hidden' : 24}
    ``ignore_keys`` : Optional[tuple[str, ...]]
        A tuple that defines the keywords that should be ignored, 
        or in other words not filterd out.
        
    Returns
    -------
    ``filtered_results`` : list[AMPC_data | MPC_data]
        Are the filtered results in a list.
    """
    def value_matches(value, filter_val):
        if isinstance(filter_val, list):
            return value in filter_val
        else:
            return value == filter_val
        
    if not or_filter_dict and not and_filter_dict:
        return results
    
    if or_filter_dict:
        filtered_results = list(filter(
            lambda x: any(value_matches(getattr(x.P, key), val) for key, val in or_filter_dict.items() if ignore_keys is None or (key not in ignore_keys)), 
            results
        ))
        orig_set = {getattr(result.P, key) for result in results for key, val in or_filter_dict.items()}
        new_set = {getattr(result.P, key) for result in filtered_results for key, val in or_filter_dict.items()}
        filter_keys = set(or_filter_dict.keys())
    else:
        filtered_results = results
        orig_set = set()
        new_set = set()
        filter_keys = set()

    if and_filter_dict:
        filtered_results = list(filter(
            lambda x: all(value_matches(getattr(x.P, key), val) for key, val in and_filter_dict.items() if ignore_keys is None or (key not in ignore_keys)), 
            filtered_results))
        orig_set.update({getattr(result.P, key) for result in results for key, val in and_filter_dict.items()})
        new_set.update({getattr(result.P, key) for result in filtered_results for key, val in and_filter_dict.items()})
        filter_keys.update(set(and_filter_dict.keys()))
        
    if not filtered_results:
        raise ValueError(f'No matching values for filter keys: {filter_keys} -> Exist in - original: {orig_set} - new: {new_set} \nGot {and_filter_dict, or_filter_dict}') 

    return filtered_results



# ------------------------------------------------------------------------------------------------------
## OTHER STUFF
# ------------------------------------------------------------------------------------------------------
def find_str_between(name: str, left: str, right: str):
    """
    Finds the ``substring`` in ``name``, which is bounded by the ``left`` and ``right`` string.

    Parameters
    ----------
    ``name`` (str):
        This string should contain the boundries and the substring.
    ``left`` (str):
        This string should be unique, it determines the left boundry of the desired substring.
    ``right`` (str):
        This string should be unique, it determines the right boundry of the desired substring.

    Returns
    ------
    ``substring`` (str): 
        The substring bounded by the provided left and right strings.  
    """
    return name[name.index(left) + len(left) : name.index(right)]


def find_already_trained_data(path: os.PathLike, n_samples: int, start = 'MPC_data_', step_end = 'steps_', samples_end = 'datapoints'):
    """
    Find already trained datasets in the given ``path`` and with the given number of samples.

    Parameters
    ----------
    ``path`` (PathLike):
        The complete path of the saved datasets.
    ``n_samples`` (int):
        The number of samples intended to be used

    Keyword parameters
    -------------------
    ``start`` (str, optional):
        The starting string of the 0th element to the element before the data to be considered, by default 'MPC_data_' 
    ``step_end`` (str, optional):
        The string marking the end of steps, by default 'steps_'
    ``samples_end`` (str, optional):
        The string marking the end of the sample count, by default 'datapoints'

    Returns
    -------
    ``already_trained_steps`` (list):
        A list containing the steps of already trained data that match the number of samples.
    """
    data_settings = []
    for f_name in os.listdir(path):
        if f_name.endswith('.csv') and f_name.startswith(start):
            steps = int(find_str_between(f_name, start, step_end))
            f_n_samples = int(find_str_between(f_name, step_end, samples_end))
            
            if n_samples == f_n_samples:
                data_settings.append(steps)
    return data_settings



def add_and_or_str(name: str, or_dict: dict | None, and_dict: dict | None):
    """
    Combines a base name with additional descriptors from provided dictionaries.

    Parameters
    ----------
    ``name`` : str
        The base name to which descriptors are added.
    ``or_dict`` : dict, optional
        Dictionary of values to be concatenated using OR logic.
        Default = None.
    ``and_dict`` : dict, optional
        Dictionary of values to be concatenated using AND logic.
        Default = None.

    Returns
    -------
    ``result`` : str
        The combined string with the base name and additional descriptors.
    """
    if or_dict is None:
        or_dict = {}
    if and_dict is None:
        and_dict = {}

    key_entries = {
        'N_NN': 'N',
        'N_MPC': 'M',
        'N_hidden': 'Nh',
        'N_hidden_end': 'Nhe',
    }

    or_list = []
    for key, val in or_dict.items():
        if isinstance(val, (list, tuple)):
            val = '_'.join(str(el) for el in val)
        or_list.append(f'{val}{key_entries[key]}')

    and_list = []
    for key, val in and_dict.items():
        if isinstance(val, (list, tuple)):
            val = '_'.join(str(el) for el in val)
        and_list.append(f'{val}{key_entries[key]}')

    return '_'.join((name, *or_list, *and_list)) if or_list or and_list else name



def merge_listdicts(*dicts: dict[str, list]) -> dict[str, list]:
    """
    Merge multiple dictionaries by concatenating lists for duplicate keys.

    Parameters
    ----------
    ``*dicts`` : dict[str, list] 
        Dictionaries to be merged.

    Returns
    -------
    ``merged_dict`` : dict[str, list]
        A single dictionary with merged values.
    """
    merged = defaultdict(list)

    for d in dicts:
        for key, value in d.items():
            if isinstance(value, list):
                merged[key].extend(value)
            else:
                merged[key].append(value)
    
    return dict(merged)