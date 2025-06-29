import psutil

from functools import wraps
from typing import Callable, Literal
from datetime import datetime



prev_fun_type = ['deleted']

def enforce_call_order(fun_type: Literal[
    'set_ocp', 'set_cost', 'set_constraints', 'set_stage_cost', 'set_terminal_cost',
    'set_solver_options', 'set_ocp_string', 'set_ocp_solver', 'deleted'
]):
    """
    This decorator ensures a specific call order of the AMPC and NH-AMPC classes. 
    If functions of these MPCs are changed the decorator need to be passed with 
    the appropriate value.

    Parameters
    ----------
    ``fun_type`` : Literal['set_ocp', 'set_cost', 'set_constraints', 'set_stage_cost', 'set_terminal_cost', 'set_solver_options', 'set_ocp_string', 'set_ocp_solver', 'deleted']
        The string to define which function type the wrapped function is. 

    Returns
    -------
    ``decorator``
        The generated decorator with its wrapper function.
    """
    expected_order = [
        'set_ocp',
        'set_cost',
        'set_constraints',
        'set_solver_options',
        'set_ocp_string',
        'set_ocp_solver',
        'deleted'
    ]
    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(*args, **kwargs):
            global prev_fun_type

            if fun_type == 'deleted':
                prev_fun_type[0] = fun_type
                return func(*args, **kwargs)

            if (prev_fun_type[0] == 'deleted') and (fun_type != 'set_ocp') or \
                    ((prev_fun_type[0] != 'deleted') and fun_type not in ['set_stage_cost', 'set_terminal_cost'] and \
                    ((expected_order.index(prev_fun_type[0]) + 1) != expected_order.index(fun_type))):
                raise ValueError(f"Function '{fun_type}' called out of order. {prev_fun_type[0]} -> {fun_type}")
            
            if fun_type in ['set_stage_cost', 'set_terminal_cost']:
                prev_fun_type[0] = 'set_cost'
            else:
                prev_fun_type[0] = fun_type

            return func(*args, **kwargs)
        return wrapper
    return decorator




def log_memory_usage(log_file_path: str):
    """
    A decorator that loggs the memory in a file before and after one function execution. 
    Percent, Total, Used, Shared, Available, Buffer and Cached Memory are stored.

    Parameters
    ----------
    ``log_file_path`` : str
        The path to the file of where the memory should be logged

    Returns
    -------
    ``decorator``
        The generated decorator with its wrapper function.
    """
    def decorator(func):
        log_memory_usage.counter = 0

        hashtag_line = '#'*80
        new_execution_msg = f'{hashtag_line}\n{datetime.now()}\n{hashtag_line}\n'
        with open(log_file_path, "a") as log_file:
            log_file.write(new_execution_msg)

        @wraps(func)
        def wrapper(*args, **kwargs):
            log_memory_usage.counter += 1

            def log_memory(stage):
                mem_info = psutil.virtual_memory()
                log_message = (
                    f"==================== {stage} (Iteration {log_memory_usage.counter}) ====================\n"
                    f"Percent used:        {mem_info.percent *100:8.2f} MB\n"
                    f"Total Memory:        {mem_info.total / (1024 ** 2):8.2f} MB\n"
                    f"Used Memory:         {mem_info.used / (1024 ** 2):8.2f} MB\n"
                    f"Shared Memory:       {mem_info.shared / (1024 ** 2):8.2f} MB\n"
                    f"Available Memory:    {mem_info.available / (1024 ** 2):8.2f} MB\n"
                    f"Buffer Memory:       {mem_info.buffers / (1024 ** 2):8.2f} MB\n"
                    f"Cached Memory:       {mem_info.cached / (1024 ** 2):8.2f} MB\n"
                )
                with open(log_file_path, "a") as log_file:
                    log_file.write(log_message)

            log_memory("Before execution")
            result = func(*args, **kwargs)
            log_memory("After execution")
            return result
        return wrapper
    return decorator