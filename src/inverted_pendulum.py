import casadi as cs
import numpy as np
from .parameters import MPC_Param

def nonlinear_pend_dynamics(P:MPC_Param) -> cs.Function:
    """
    Returns a function handle that computes the nonlinear dynamics of a pendulum on a cart system, based on the given
    physical parameters.

    Args:
    - P: a dataclass object with the following attributes:
        - nx (int): the number of states in the system
        - nu (int): the number of inputs to the system
        - l (float): the length of the pendulum arm
        - m (float): the mass of the pendulum
        - M (float): the mass of the cart
        - g (float): the gravitational constant

    Returns:
    - f: a CasADi Function object that takes as input two vectors x and u, with lengths nx and nu respectively, and
    returns a vector of derivatives that describe the nonlinear dynamics of the system. The elements of x and u
    correspond to the following variables:
        - x[0]: the position of the cart along the x-axis
        - x[1]: the angle between the pendulum arm and the vertical axis, measured in radians
        - x[2]: the velocity of the cart along the x-axis
        - x[3]: the angular velocity of the pendulum arm
        - u[0]: the force applied to the cart
    """
    x,u = cs.MX.sym('x',P.nx,1),cs.MX.sym('u',P.nu,1)

    ode = cs.vertcat(x[2],
                     x[3],
                     (-P.l*P.m*cs.sin(x[1])*x[3]**2 + u + P.g*P.m*cs.cos(x[1])*cs.sin(x[1])  ) / (P.M+P.m-P.m*cs.cos(x[1])**2),
                     (-P.l*P.m*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + P.g*P.m*cs.sin(x[1]) + P.M*P.g*cs.sin(x[1])) / (P.l*(P.M+P.m-P.m*cs.cos(x[1])**2))
                    )

    # function for ode
    f = cs.Function('f',[x,u],[ode],['x','u'],['ode'])
    return f


# def F_model_RK4(func,P):
def F_model_RK4(func:cs.Function, P:MPC_Param) -> cs.Function:
    """
    Returns a function handle that computes the state of the nonlinear pendulum on a cart system at the next time step,
    using a fourth-order Runge-Kutta integration scheme.

    Args:
    - func: a CasADi Function object that takes as input two vectors x and u, with lengths nx and nu respectively, and
    returns a vector of derivatives that describe the nonlinear dynamics of the system.
    - P: a dataclass object with the following attributes:
        - nx (int): the number of states in the system
        - nu (int): the number of inputs to the system
        - Ts (float): the duration of each time step

    Returns:
    - F_model: a CasADi Function object that takes as input two vectors x and u, with lengths nx and nu respectively, and
    returns a vector of the same length as x, containing the state of the system at the next time step.
    """
    x,u = cs.MX.sym('x',P.nx,1),cs.MX.sym('u',P.nu,1)
    dae = {'x':x,'p':u,'ode':func(x,u)}

    intg = cs.integrator('intg','rk',dae,{'tf':P.Ts, 'simplify':True, 'number_of_finite_elements':4})

    res = intg(x0=x,p=u)
    x_next = res['xf']
    F_model = cs.Function('F_model',[x,u],[x_next])
    return F_model
