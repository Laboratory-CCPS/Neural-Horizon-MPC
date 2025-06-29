import numpy as np

from typing import Optional, Literal, Callable, Any
from dataclasses import fields
from .parameters import AMPC_Param, NH_AMPC_Param
from .mpc_dataclass import MPC_data, AMPC_data
from .plotting import find_Ts



def get_mean_of_results(
        MPC_results: list[AMPC_data | MPC_data], 
        cost_fun: Optional[Callable] = None, 
        time_fun: Optional[Callable] = None, 
        keep_fields: Optional[list[Literal['acados_name', 'name']]] = None
    ) -> AMPC_data | MPC_data:
    """
    Calculate the mean of results across multiple runs of one acados option combination.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        A list of AMPC_data or MPC_data instances that should be averaged.
    ``cost_fun`` : Callable, optional
        A function to calculate the cost average.
        Default = None.
    ``time_fun`` : Callable, optional
        A function to calculate the time average.
        Default = None.
    ``keep_fields`` : list[Literal['acados_name', 'name']], optional
        A list of field names to keep from the first MPC_results entry.
        Default = None.

    Returns
    -------
    ``MPC_means`` : AMPC_data | MPC_data
        A class instance of AMPC_data or MPC_data representing the means or medians of the input results.
    """
    if keep_fields is None:
        keep_fields = []

    
    next_results = next(iter(MPC_results))
    MPC_means = AMPC_data(P=next_results.P, acados_options=next_results.acados_options) \
        if type(next_results) == AMPC_data else MPC_data(P=next_results.P)
    
    if type(MPC_means.P) == NH_AMPC_Param:
        MPC_means.P.V_NN == None
        MPC_means.P._set_new_NN_name()
        MPC_means.P._set_new_pruned_NN_name()

    for field in fields(next_results):
        field_name = field.name

        if field_name in ['P', 'acados_name', 'acados_options', 'name', '_is_frozen'] and not field_name in keep_fields:
            continue
        elif field_name in keep_fields:
            setattr(MPC_means, field_name, getattr(MPC_results[0], field_name))
        else:
            if cost_fun is not None and (field_name == 'Cost'):
                m_fun = cost_fun
            elif time_fun is not None and ('Time' in field_name):
                m_fun = time_fun
            else:
                m_fun = np.nanmean
            all_runs_value = [getattr(one_run_results, field_name) for one_run_results in MPC_results]

            try:
                setattr(MPC_means, field_name, m_fun(np.array(all_runs_value), axis=0))
            except ValueError as e:
                unique_shapes = {vals.shape for vals in all_runs_value}
                raise ValueError(f'{e}\nIn field: {field_name} -> unique shapes: {list(unique_shapes)}\nWith function {m_fun.__name__}')

    MPC_means.freeze()
    return MPC_means


# ------------------------------------------------------------------------------------------------------
## MINIMAL VALUES
# ------------------------------------------------------------------------------------------------------
def _get_min_max_mean_results(
        MPC_results: list[AMPC_data | MPC_data], 
        find_fun: Callable[[np.ndarray | object], float | object] = np.max,
        of_attribute: Callable[[AMPC_data | MPC_data], np.ndarray | object] = lambda x: x.U, # for Input 
    ):
    """
    Calculates the results based on the provided function for each attribute in the MPC results.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        The list of MPC result instances to be evaluated.
    ``find_fun`` : Callable[[np.ndarray | object], float | object], optional
        The function used to compute the desired value (e.g., np.max, np.min, np.mean). Default is np.max.
    ``of_attribute`` : Callable[[AMPC_data | MPC_data], np.ndarray | object], optional
        A function to extract the attribute to be evaluated from each MPC result. Default is lambda x: x.U.

    Returns
    ------
    ``results`` : list
        A list of tuples, where each tuple contains the computed value and the corresponding MPC data.
    """
    return [(find_fun(of_attribute(results)), results) for results in MPC_results]


def get_minimum_of_find_fun_results(
        MPC_results: list[AMPC_data | MPC_data], 
        find_fun: Callable[[np.ndarray | object], object] = np.max,
        of_attribute: Callable[[AMPC_data | MPC_data], np.ndarray | object] = lambda x: x.U, # for Input 
        error_msg: str = 'Calculated minimal result is not right!'
    ):
    """
    Calculates the minimum of the results computed by a specified function across all MPC instances.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        The list of MPC result instances to be evaluated.
    ``find_fun`` : Callable[[np.ndarray | object], object], optional
        The function used to compute the desired value (e.g., np.max, np.min, np.mean). Default is np.max.
    ``of_attribute`` : Callable[[AMPC_data | MPC_data], np.ndarray | object], optional
        A function to extract the attribute to be evaluated from each MPC result. Default is lambda x: x.U.
    ``error_msg`` : str, optional
        The error message displayed if the minimum result calculation is incorrect. Default is 'Calculated minimal result is not right!'.

    Returns
    ------
    ``minimal_result`` : float
        The minimum value of the specified result across all MPC results.
    ``minimal_result_options`` : list
        A list of acados options that achieve the minimum value of the specified result.
    """
    values_and_results = _get_min_max_mean_results(MPC_results, find_fun, of_attribute)
    values_and_results = list(filter(lambda x: x[0] != None, values_and_results))
    minimal_value_with_result = min(values_and_results, key=lambda x: x[0])
    minimal_value = minimal_value_with_result[0]

    minimal_results = [minimal_value_with_result[1]]
    for (value, results) in values_and_results:
        if minimal_value == value:
            minimal_results.append(results)
        assert minimal_value <= value or value != np.nan, f'{error_msg} {minimal_value} <= {value}'
    return minimal_value, minimal_results



def get_minimal_settling_Time(MPC_results: list[AMPC_data | MPC_data]):
    """
    Calculates the minimal settling time from a list of MPC results.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        The list of MPC result instances, each containing settling time information.

    Returns
    ------
    ``minimal_settling_time`` : float
        The minimal settling time found across all MPC results.
    ``minimal_settling_time_options`` : list
        A list of acados options that achieve the minimal settling time.
    """
    return get_minimum_of_find_fun_results(MPC_results, find_Ts, lambda x: x)



def get_minimal_mean_time(MPC_results: list[AMPC_data | MPC_data]):
    """
    Calculates the minimal mean calculation time from a list of MPC results.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        The list of MPC result instances, each containing mean calculation time information.

    Returns
    ------
    ``minimal_mean_time`` : float
        The minimal mean calculation time found across all MPC results.
    ``minimal_mean_time_options`` : list
        A list of acados options that achieve the minimal mean calculation time.
    """
    return get_minimum_of_find_fun_results(MPC_results, find_fun=np.mean, of_attribute=lambda x: x.Time)



def get_minimal_max_input_diff(MPC_diff: list[AMPC_data | MPC_data]):
    """
    Calculates the minimal maximum input difference from a list of MPC differences.

    Parameters
    ----------
    ``MPC_diff`` : list[AMPC_data | MPC_data]
        The list of MPC difference instances, each containing input difference information.

    Returns
    ------
    ``minimal_maximum_input_difference`` : float
        The minimal maximum input difference found across all MPC difference results.
    ``minimal_maximum_input_difference_options`` : list
        A list of acados options that achieve the minimal maximum input difference.
    """
    return get_minimum_of_find_fun_results(MPC_diff, np.max, lambda x: x.U)



# ------------------------------------------------------------------------------------------------------
## DIFFERENCES
# ------------------------------------------------------------------------------------------------------
def get_MPC_diffs(
        base_mpc_results: AMPC_data | MPC_data, 
        MPC_res: list[AMPC_data | MPC_data]
    ):
    """
    Calculates the differences of all controllers compared to a base comparison controller.

    Parameters
    ----------
    ``base_mpc_results`` : AMPC_data | MPC_data
        The base results of the comparison controller.
    ``MPC_res`` : list[AMPC_data | MPC_data]
        The results of MPC's with all parameters, inputs, states, its trajectories, and calculation time for each step.

    Returns
    ------
    ``MPC_diffs`` : list
        A list of differences of all controllers relative to the base comparison controller, structured the same as MPC_res.
    """
    MPC_diffs = []
    # subtract the comparison mpc costs of every other mpc costs
    for cresults in MPC_res:
        if isinstance(base_mpc_results, AMPC_data) and isinstance(cresults, AMPC_data):
            cdiffs = AMPC_data(cresults.P)
        else: 
            cdiffs = MPC_data(cresults.P)

        cdiffs.X = cresults.X - base_mpc_results.X
        cdiffs.U = cresults.U - base_mpc_results.U
        cdiffs.X_traj = cresults.X_traj - base_mpc_results.X_traj
        cdiffs.U_traj = cresults.U_traj - base_mpc_results.U_traj
        cdiffs.Time = cresults.Time - base_mpc_results.Time
        cdiffs.Cost = cresults.Cost - base_mpc_results.Cost

        if isinstance(cdiffs, AMPC_data):
            cdiffs.Acados_Time = cresults.Acados_Time - base_mpc_results.Acados_Time
            cdiffs.Prep_Time = cresults.Prep_Time - base_mpc_results.Prep_Time
            cdiffs.Fb_Time = cresults.Fb_Time - base_mpc_results.Fb_Time

            cdiffs.Iterations = cresults.Iterations - base_mpc_results.Iterations
            cdiffs.Prep_Iterations = cresults.Prep_Iterations - base_mpc_results.Prep_Iterations
            cdiffs.Fb_Iterations = cresults.Fb_Iterations - base_mpc_results.Fb_Iterations
            
        cdiffs.freeze()
        MPC_diffs.append(cdiffs)
    return MPC_diffs
