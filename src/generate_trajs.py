import os
import torch

from .utils import save_results
from .parameters import NH_AMPC_Param, AMPC_Param
from .mpc_classes_acados import get_AMPC_trajectory, NH_AMPC_class, AMPC_class
from .inverted_pendulum_acados import acados_model_RK4, nonlinear_pend_dynamics_acados
from .neural_horizon import load_NN
from .decorators import log_memory_usage




@log_memory_usage('Temp/log_nh_ampc_mem.txt')
def generate_NH_AMPC_trajs(
        nh_ampc_param: NH_AMPC_Param,
        solver_option: tuple[str, dict], 
        save_dir: str, 
        nn_dir: str, 
        ds_dir: str, 
        device: torch.device,
        dtype: torch.NumberType,
        always_overwrite: bool = True,
        **ampc_class_kwargs
    ):
    """
    Generates the trajectory for a given NH_AMPC_Param with the given solver options.
    Loggs the memory usage before and after execution in 'Temp/log_nh_ampc_mem.txt'.

    Parameters
    ----------
    ``nh_ampc_param`` : NH_AMPC_Param
        A dataclass with relevant NH-AMPC parameters. 
         Including the dataset name as well as neural network name...
    ``solver_option`` : tuple[str, dict]
        A tuple where the first item is a string defining the MPC acados name 
        and the second item is a dictionary with solver options. <br>
        e.g.: 
            - 'qp_solver': 'FULL_CONDENSING_HPIPM',
            - 'integrator_type': 'DISCRETE',
            - 'nlp_solver_type': 'SQP_RTI',
            - 'as_rti_iter': 3,
            - 'as_rti_level': 3,
            - 'nlp_solver_tol_stat': 1e-6,
            - 'nlp_solver_max_iter': 3
    ``save_dir`` : str
        The directory where the results should be stored.
    ``nn_dir`` : str
        The directory to the trained neural networks.
    ``ds_dir`` : str
        The directory to the datasets.
    ``device`` : torch.device
        The device cpu or cuda where the neural network should be loaded in.
    ``dtype`` : torch.NumberType
        The datatype for the neural networks 
    ``always_overwrite`` : bool
        Specifier to say always overwrite the results if alredy existent.
         Default = True
    ``**ampc_class_kwargs``
        Keyword-arguments directly passed to the NH_AMPC_class.

    Returns
    -------
    ``results`` : AMPC_data
        The results containing the trajectories and timings. 
    """
    cname, coptions = solver_option
    results_filename = f'NH_AMPC_results_{cname}_{nh_ampc_param.Pruned_NN_name if nh_ampc_param.N_hidden_end is not None else nh_ampc_param.NN_name}'

    NN_fc = load_NN(nh_ampc_param, nn_dir, ds_dir, device, dtype)
    NN_fc.evaluate_NN(os.path.join(ds_dir, nh_ampc_param.test_DS_name))
    
    model = acados_model_RK4(nonlinear_pend_dynamics_acados(nh_ampc_param), nh_ampc_param)
    cAMPC = NH_AMPC_class(
        model,
        NN_fc, 
        solver_options=coptions, 
        horizon_name=nh_ampc_param.param_name, 
        acados_name=cname, 
        **ampc_class_kwargs
    )
    try:
        results = get_AMPC_trajectory(cAMPC, show_tqdm=False, verbose=False)
    except Exception as e:
        raise e
    finally:
        cAMPC.cleanup()
        del cAMPC

    save_results(os.path.join(save_dir, results_filename), results, always_overwrite=always_overwrite)
    return results



@log_memory_usage('Temp/log_ampc_mem.txt')
def generate_AMPC_trajs(
        ampc_param: AMPC_Param, 
        solver_option: tuple[str, dict], 
        save_dir: str,
        always_overwrite: bool = True,
        **ampc_class_kwargs
    ):
    """
    Generates the trajectory for a given AMPC_Param with the given solver options.
    Loggs the memory usage before and after execution in 'Temp/log_ampc_mem.txt'.

    Parameters
    ----------
    ``ampc_param`` : AMPC_Param
        A dataclass with relevant AMPC parameters.
    ``solver_option`` : tuple[str, dict]
        A tuple where the first item is a string defining the MPC acados name 
         and the second item is a dictionary with solver options.<br>
        e.g.: 
            - 'qp_solver': 'FULL_CONDENSING_HPIPM',
            - 'integrator_type': 'DISCRETE',
            - 'nlp_solver_type': 'SQP_RTI',
            - 'as_rti_iter': 3,
            - 'as_rti_level': 3,
            - 'nlp_solver_tol_stat': 1e-6,
            - 'nlp_solver_max_iter': 3
    ``save_dir`` : str
        The directory where the results should be stored.
    ``always_overwrite`` : bool
        Specifier to say always overwrite the results if alredy existent.
         Default = True
    ``**ampc_class_kwargs``
        Keyword-arguments directly passed to the AMPC_class.
    
    Returns
    -------
    ``results`` : AMPC_data
        The results containing the trajectories and timings. 
    """
    cname, coptions = solver_option
    results_filename = f'AMPC_results_{cname}_{ampc_param.param_name}.ph'

    model = acados_model_RK4(nonlinear_pend_dynamics_acados(ampc_param), ampc_param)
    cAMPC = AMPC_class(
        model,
        ampc_param, 
        solver_options=coptions, 
        horizon_name=ampc_param.param_name, 
        acados_name=cname, 
        ignore_status_errors={0, 5}, 
        **ampc_class_kwargs
    )
    try:
        results = get_AMPC_trajectory(cAMPC, show_tqdm=False, verbose=False)
    except Exception as e:
        raise e
    finally:
        cAMPC.cleanup()
        del cAMPC
    
    save_results(os.path.join(save_dir, results_filename), results, always_overwrite=always_overwrite)
    return results