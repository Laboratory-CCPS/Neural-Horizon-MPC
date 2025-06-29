import casadi as cs

from acados_template import AcadosModel
from .parameters import AMPC_Param, NH_AMPC_Param



def nonlinear_pend_dynamics_acados(P: AMPC_Param | NH_AMPC_Param) -> AcadosModel:
    """
    Creates an AcadosModel with the nonlinear dynamics of a pendulum on a cart system, based on the given
    physical parameters.

    Parameters
    ----------
    ``P`` : AMPC_Param | NH_AMPC_Param
        A dataclass object specifying the parameters for the Acados Model.

    Returns
    -------
    ``model`` : AcadosModel
        An AcadosModel containing explicite and implicite expressions. It defines the two vectors x and u, 
        with lengths nx and nu respectively. The elements of x and u
        correspond to the following variables:
            - x[0]: the position of the cart along the x-axis
            - x[1]: the angle between the pendulum arm and the vertical axis, measured in radians
            - x[2]: the velocity of the cart along the x-axis
            - x[3]: the angular velocity of the pendulum arm
            - u[0]: the force applied to the cart
    """
    p = []

    x,u = cs.MX.sym('x', P.nx, 1), cs.MX.sym('u', P.nu, 1)
    xdot = cs.MX.sym('x_dot', P.nx, 1)

    f_expl = cs.vertcat(
        x[2],
        x[3],
        (-P.l*P.m*cs.sin(x[1])*x[3]**2 + u + P.g*P.m*cs.cos(x[1])*cs.sin(x[1])) / (P.M+P.m-P.m*cs.cos(x[1])**2),
        (-P.l*P.m*cs.cos(x[1])*cs.sin(x[1])*x[3]**2 + u*cs.cos(x[1]) + P.g*P.m*cs.sin(x[1]) + P.M*P.g*cs.sin(x[1])) / (P.l*(P.M+P.m-P.m*cs.cos(x[1])**2))
    )
    
    f_impl = xdot - f_expl
    
    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = 'pendulum'

    return model


def acados_model_RK4(model: AcadosModel, P: AMPC_Param | NH_AMPC_Param):
    """
    Extends the given ``model`` with a discrete RK4 expression.

    Parameters
    ----------
    ``model`` : AcadosModel
        An AcadosModel containing explicite and implicite expressions.
    ``P`` : AMPC_Param | NH_AMPC_Param
        The MPC_Param dataclass containing the MPC problem parameters.

    Returns
    -------
    ``model`` : AcadosModel
        Extended AcadosModel with a discrete RK4 expression. 
    """

    x = model.x
    u = model.u

    ode = cs.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+P.Ts/2*k1,u)
    k3 = ode(x+P.Ts/2*k2,u)
    k4 = ode(x+P.Ts*k3,  u)
    xf = x + P.Ts/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    return model