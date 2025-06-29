import numpy as np
import pandas as pd


from itertools import chain
from scipy.stats import norm
from typing import Callable, Optional, Literal, Iterable
from types import MappingProxyType
from collections import defaultdict, Counter
from math import pi
from bokeh.plotting import figure
from bokeh.models import (
    BoxAnnotation, Legend, GridPlot, Column, Row, AdaptiveTicker, PrintfTickFormatter,
    Whisker, ColumnDataSource, HoverTool, LegendItem, FactorRange, FixedTicker, GlyphRenderer,
    ColorBar, Range1d, Span, CustomJS, Select, RadioButtonGroup, ButtonLike
)
from bokeh.layouts import gridplot, column
from bokeh.transform import linear_cmap, factor_cmap, factor_mark, jitter, CategoricalColorMapper
from bokeh.palettes import Turbo256, Category20, Category10_10, Viridis10


from .mpc_dataclass import AMPC_data, MPC_data, dataclass_group_by
from .utils import merge_listdicts




latex_labels = MappingProxyType({
    'N_MPC': r'$$M$$',
    'N_NN': r'$$N_{NN}$$',
    'N_hidden': r'$$n_{hidden}$$',
    'N_hidden_end': r'$$n_{hidden}$$',
    'Mean_Time': r'$$\bar{t}_{sol} \, [\mathrm{s}]$$',
    'Median_Time': r'$$\tilde{t}_{sol} \, [\mathrm{s}]$$',
    'Cost': r'Cost',
    'R2_score': r'$$R^2$$'
})


def latex_label_with_unit(label: str, unit: Optional[str] = None):
    """
    Generates a LaTeX-formatted label with an optional unit.

    Parameters
    ----------
    ``label`` : str
        The label to be formatted in LaTeX.
    ``unit`` : Optional[str], optional
        The unit to be appended to the label. If None, no unit is added. Default is None.

    Returns
    -------
    ``latex_label`` : str
        The LaTeX-formatted label with the unit if provided.
    """
    latex_label = latex_labels[label]
    left_bracket_index = latex_label.find('[')
    right_bracket_index = latex_label.find(']')
    if unit is None:
        return latex_label
    elif left_bracket_index != -1 and right_bracket_index != -1:
        return latex_label.replace(latex_label[left_bracket_index:right_bracket_index+1], fr'[\mathrm{{{unit}}}]')
    else:
        return fr'{latex_label[:-2]} \, [\mathrm{{{unit}}}]$$'



def exclude_buttons_figure(p: figure | Column | Row | GridPlot):
    """
    Removes button-like renderers and toolbars from a Bokeh figure or layout.

    Parameters
    ----------
    ``p`` : figure | Column | Row | GridPlot
        The Bokeh figure or layout to clean.

    Returns
    -------
    ``clean_p`` : figure | Column | Row | GridPlot
        The cleaned Bokeh figure or layout without button-like renderers and toolbars.
    """
    def clean_item(item):
        if isinstance(item, (Column, Row)):
            new_childs = []
            for it in item.children:
                new_item = clean_item(it)
                if new_item is not None:
                    new_childs.append(new_item)
            item.children = new_childs
            return item

        elif isinstance(item, GridPlot):
            new_childs = []
            for it in item.children:
                new_item = clean_item(it[0])
                if new_item is not None:
                    new_childs.append((new_item, *it[1:]))
            item.children = new_childs
            item.toolbar_location = None
            item.toolbar.logo = None
            return item
        
        elif isinstance(item, figure):
            new_renderers = [r for r in item.renderers if not isinstance(r, ButtonLike)]
            item.renderers = new_renderers
            item.toolbar_location = None
            item.toolbar.logo = None
            return item
        
        elif isinstance(item, (list, tuple)):
            new_items = [] 
            for it in item:
                new_it = clean_item(it)
                if new_it is not None:
                    new_items.append(new_it)
            return new_items 
            
    p = clean_item(p)

    return p


def pt2px_height(pt: int):
    """
    Converts a height value from points to pixels.

    Parameters
    ----------
    ``pt`` : int
        The height in points to be converted.

    Returns
    -------
    ``px_height`` : int
        The height in pixels.
    """
    return int(np.round(pt) * 1.52)


def get_legend_height(legend: Legend):
    """
    Calculates the height of a Bokeh legend in pixels.

    Parameters
    ----------
    ``legend`` : Legend
        The Bokeh legend for which the height is calculated.

    Returns
    -------
    ``height`` : int
        The height of the legend in pixels.
    """
    label_font_size = pt2px_height(int(legend.label_text_font_size[:-2]))
    title_font_size = pt2px_height(int(legend.title_text_font_size[:-2]))
    num_items = len(legend.items)
    if legend.ncols is None or legend.ncols == 1:
        # If ncols is not set or 1, calculate height for a single column
        height = num_items * max(label_font_size, legend.label_height, legend.glyph_height) + (num_items - 1) * legend.spacing
    else:
        # If ncols is set, calculate height based on the number of columns
        num_rows = (num_items + legend.ncols - 1) // legend.ncols
        height = num_rows * max(label_font_size, legend.label_height, legend.glyph_height) + (num_rows - 1) * legend.spacing

    height += 2 * (legend.padding + legend.margin)

    if legend.title:
        height += title_font_size + legend.spacing 

    return height



def get_legend_item_widths(legend: Legend):
    """
    Calculates the widths of items in a Bokeh legend.

    Parameters
    ----------
    ``legend`` : Legend
        The Bokeh legend for which the item widths are calculated.

    Returns
    -------
    ``widths`` : list of int
        A list of widths for each legend item in pixels.
    """
    widths = []
    legend_text_size = pt2px_height(int(legend.label_text_font_size[:-2]))
    for item in legend.items:
        label = item.label['value']
        widths.append(len(label) * legend_text_size)  # Estimate width by character length, adjust multiplier as needed
    return widths



def calculate_ncols(plot_width, legend: Legend, max_width_factor: float = 0.85):
    """
    Calculates the number of columns for a Bokeh legend to fit within the plot width.

    Parameters
    ----------
    ``plot_width`` : int
        The width of the plot in pixels.
    ``legend`` : Legend
        The Bokeh legend for which the number of columns is calculated.
    ``max_width_factor`` : float, optional
        The maximum width factor of the plot to be occupied by the legend.
         Default = 0.85.

    Returns
    -------
    ``ncols`` : int
        The number of columns for the legend.
    """
    item_widths = get_legend_item_widths(legend)
    available_width = int(plot_width * max_width_factor - 2 * (legend.padding + legend.margin))# Subtract a margin from the plot width for padding
    ncols = 1
    current_width = 0
    max_width = max(*item_widths, legend.label_width)
    for _ in range(len(item_widths)):
        add_width = max_width + legend.glyph_width + legend.spacing + legend.label_standoff
        if (current_width + add_width) < available_width:
            ncols += 1
            current_width += add_width
        else:
            break
    
    ncols = min(len(item_widths), ncols)
    return ncols



def get_figure_size(latex_text_width_pt = 497.92325, fraction: float = 1., ratio: Optional[float]=None):
    """
    Calculate the width and height of a figure based on a given fraction of the LaTeX text width.
    The height is determined using the golden ratio.

    Parameters:
    ``fraction``  : float, optional
        Fraction of the LaTeX text width (e.g., 0.5 for half the width)

    Returns:
    ``width, height`` : tuple[int, int] 
        Width and height of the figure in pixels
    """
    # Convert points to pixels (1 pt = 1.333 pixels)
    latex_text_width_px = latex_text_width_pt * 1.333

    # Calculate width and height based on the golden ratio
    width = latex_text_width_px * fraction

    if ratio is None:
        golden_ratio = (1 + 5 ** 0.5) / 2  # Approx 1.618
        height = width / golden_ratio
    else:
        height = width / ratio

    return int(width), int(height)



def style_single_figure(
        p: figure,
        line_color = 'black',
        text_color = 'black',
        font = 'serif',
        tick_font_size = '10pt',
        label_font_size = '12pt',
        line_width = 2,
        dot_size = 4,
        output_backend: Optional[Literal['svg', 'html']] = None,
        move_legend_top: bool = True,
    ):
    """
    Styles a Bokeh figure with specified aesthetic parameters.

    Parameters
    ----------
    ``p`` : figure
        The Bokeh figure to style.
    ``line_color`` : str, optional
        Color of the axis lines, ticks, and legend border. Default is 'black'.
    ``text_color`` : str, optional
        Color of the axis labels, tick labels, and legend text. Default is 'black'.
    ``font`` : str, optional
        Font style for axis labels, tick labels, and legend text. Default is 'serif'.
    ``tick_font_size`` : str, optional
        Font size for the axis tick labels. Default is '10pt'.
    ``label_font_size`` : str, optional
        Font size for the axis labels. Default is '12pt'.
    ``line_width`` : int, optional
        Width of the lines in the figure. 
         Default = 2.
    ``dot_size`` : int, optional
        Size of the dots in the figure. Default is 4.
    ``output_backend`` : Optional[Literal['svg', 'html']], optional
        Output backend for the figure rendering. Default is None.
    ``move_legend_top`` : bool, optional
        If True, moves the legend to the top of the figure. Default is True.
    """
    if output_backend is not None:
        p.output_backend = output_backend

    # Set background and grid style
    p.background_fill_color = "#fafafa"
    p.xgrid.grid_line_color = "#e0e0e0"
    p.ygrid.grid_line_color = "#e0e0e0"
    p.xgrid.grid_line_dash = [6, 4]
    p.ygrid.grid_line_dash = [6, 4]

    # Set font styles
    p.title.text_font = font
    p.xaxis.axis_label_text_font = font
    p.yaxis.axis_label_text_font = font
    p.xaxis.major_label_text_font = font
    p.yaxis.major_label_text_font = font
    p.axis.major_label_text_font_size = tick_font_size
    p.axis.axis_label_text_font_size = label_font_size
    p.axis.axis_line_color = line_color
    p.axis.major_tick_line_color = line_color
    p.axis.minor_tick_line_color = line_color
    p.axis.major_label_text_color = text_color
    p.axis.axis_label_text_color = text_color
    p.outline_line_color = None

    if hasattr(p.xaxis, 'group_text_font_size'):
        p.xaxis.group_text_font_style = 'normal' 
        p.xaxis.group_text_color = 'black'
        p.xaxis.group_text_font = font
        p.xaxis.group_text_font_size = tick_font_size

        p.xaxis.major_label_text_font_size = '0pt'

    # Set Line styles
    for renderer in p.renderers:
        if hasattr(renderer, 'glyph'):
            if hasattr(renderer.glyph, 'line_width'):
                renderer.glyph.line_width = line_width # Adjust the thickness as needed
            if hasattr(renderer.glyph, 'size'):
                renderer.glyph.size = dot_size

    # Customize legend
    if len(p.legend) > 0:
        legend_items = [LegendItem(label=item.label, renderers=item.renderers) for item in p.legend.items]
        new_legend = Legend(items=legend_items, location="top_left", orientation="horizontal")

        ## Customize legend
        new_legend.title = p.legend.title
        new_legend.title_text_font = font
        new_legend.title_text_font_style = "bold"
        new_legend.label_text_font = font
        new_legend.label_text_font_size = tick_font_size
        new_legend.label_text_color = text_color
        new_legend.border_line_color = line_color
        new_legend.border_line_width = 1
        new_legend.border_line_alpha = 0.8
        new_legend.background_fill_color = "#f0f0f0"
        new_legend.background_fill_alpha = 0.8

        new_legend.spacing = 5
        # new_legend.label_width = 100
        
        new_legend.ncols = calculate_ncols(p.width, new_legend)

        # Remove the original legend
        p.legend.visible = False

        p.height += get_legend_height(new_legend) - get_legend_height(p.legend)

        # Add the new legend above the plot
        p.add_layout(new_legend, 'above' if move_legend_top else 'below')

    cbar: ColorBar = p.select_one(ColorBar)
    if cbar is not None:
        cbar.major_label_text_font = font
        cbar.title_text_font = font
        cbar.major_label_text_color = text_color
        cbar.title_text_color = text_color
        cbar.major_label_text_font_size = tick_font_size
        cbar.title_text_font_size = label_font_size
        cbar.bar_line_color = line_color
        cbar.major_tick_line_color = line_color

        


def set_figure_to_default_latex(p: figure | Column | Row | GridPlot | list, **kwargs) -> None:
    """
    Applies default LaTeX styling to a Bokeh figure or layout, including nested layouts.

    Parameters
    ----------
    ``p`` : figure | Column | Row | GridPlot | list
        The Bokeh figure or layout to style. Can also be a list of figures or layouts.
    ``**kwargs`` : optional
        Additional keyword arguments passed to the `style_single_figure` function.
    """
    # Check the type of p and apply styling appropriately
    if isinstance(p, figure):
        style_single_figure(p, **kwargs)
    elif isinstance(p, (Column, Row)):
        for p_child in p.children:
            set_figure_to_default_latex(p_child, **kwargs)
    elif isinstance(p, GridPlot):
        for i, item in enumerate(p.children):
            p_child = item[0]
            set_figure_to_default_latex(p_child, **kwargs)

            if i < len(p.children)-1 and hasattr(p_child, 'xaxis') and hasattr(p_child, 'yaxis') and type(p_child) != Legend and type(p_child) != ColorBar:
                p_child.height -= pt2px_height(int(p_child.xaxis.major_label_text_font_size[:-2]))
                p_child.xaxis.major_tick_line_color = None
                p_child.xaxis.minor_tick_line_color = None
                p_child.xaxis.major_label_text_font_size = '0pt'

    elif isinstance(p, list):  # Handle lists of figures
        for item in p:
            set_figure_to_default_latex(item, **kwargs)




def plot_line(
        fig: figure, 
        x_data: list, 
        y_data: list, 
        line_alpha: float, 
        line_width: int, 
        line_dash: str, 
        color: str, 
        hover_tool: bool
    ) -> GlyphRenderer:
    """
    Plots a line on a Bokeh figure with optional hover tool.

    Parameters
    ----------
    ``fig`` : figure
        The Bokeh figure to plot the line on.
    ``x_data`` : list or array-like
        The x-coordinates of the line.
    ``y_data`` : list or array-like
        The y-coordinates of the line.
    ``line_alpha`` : float
        The transparency level of the line.
    ``line_width`` : int
        The width of the line.
    ``line_dash`` : str
        The dash pattern of the line.
    ``color`` : str
        The color of the line.
    ``hover_tool`` : bool
        If True, adds a hover tool to display the line's coordinates.

    Returns
    -------
    ``handle`` : GlyphRenderer
        The renderer for the plotted line.
    """
    source = ColumnDataSource(data=dict(x=x_data, y=y_data))
    handle = fig.line(
        x='x', y='y', source=source, line_alpha=line_alpha, line_width=line_width,
        line_dash=line_dash, color=color
    )
    if hover_tool:
        hover = HoverTool(tooltips=[("x, y", "@x, @y")], mode='mouse', renderers=[handle])
        fig.add_tools(hover)
    return handle



def plot_MPC_results(
    MPC_results: list[AMPC_data | MPC_data],
    time_type: Optional[Literal['acados', 'python']] = 'acados', 
    time_scale: str = 'log',
    additional_plots: Optional[list[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']]] = None,
    additional_plots_options: Optional[dict[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations'], dict]] = None,  
    group_by: Callable[[AMPC_data | MPC_data], str] = lambda x: x.name,
    background_fill_color: str = "#FAFAFA",
    bnd_color: str = '#D55E00',
    width: int = 1200,
    height: int = 200,
    theta_bnd: float | Iterable[float] = 1.5*np.pi,
    plot_mpc_trajectories = False,
    plot_Ts: bool = True,
    activate_hover: bool = False,
    threshold: float = None,
    xbnd: Optional[float] = None,
    legend_title: Optional[str] = None,
    legend_cols: Optional[int] = None,
    # parameters of the plots
    cols = Category10_10,
    dash: list = ['solid','solid','solid','dashed','dotdash'],
    thickness: list = [8,4,4,3,3],
    alpha: list = [.75,.5,1,1,1],
    latex_style: bool = False,
):
    """
    Generates a Bokeh plot of state and input trajectories for multiple MPC results.

    Parameters
    ----------
    ``MPC_results`` : list[AMPC_data | MPC_data]
        List of MPC result data objects containing simulation and trajectory data.
    ``time_type`` : Literal['acados', 'python'], optional
        The time data source for the solver iteration time plot. 
         Default = 'acados'.
    ``additional_plots`` : list[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']], optional
        List of additional plots to include. 
         Default = None.
    ``additional_plots_options`` : dict[Literal['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations'], dict], optional
        Dictionary of additional plot options. 
         Default = None.
    ``group_by`` : Callable[[AMPC_data | MPC_data], str], optional
        Function to group MPC results by a specific attribute. 
         Default = lambda x: x.name.
    ``plot_mpc_trajectories`` : bool, optional
        Whether to plot MPC predicted trajectories. 
         Default = False.
    ``plot_Ts`` : bool, optional
        Whether to plot solver convergence times. 
         Default = True.
    ``activate_hover`` : bool, optional
        Whether to activate hover tools. 
         Default = False.
    ``threshold`` : float, optional
        Threshold for solver iteration time plot. 
         Default = None.
    ``legend_title`` : str, optional
        Title for the legend. 
         Default = None.
    ``width`` : int, optional
        Width of each plot. 
         Default = 1200.
    ``height`` : int, optional
        Height of each plot. 
         Default = 200.
    ``theta_bnd`` : float | Iterable[float], optional
        Boundary for the angle theta in radians. 
         Default = 1.5*np.pi.
    ``xbnd`` : float, optional
        Boundary for the x-axis. 
         Default = None.
    ``time_scale`` : str, optional
        The scaling of the y-axis for the solver iteration time plot. Can be 'log' or 'linear'. 
         Default = 'log'.
    ``background_fill_color`` : str, optional
        Background color of the plots. 
         Default = "#FAFAFA".
    ``bnd_color`` : str, optional
        Color for shading areas outside state boundaries. 
         Default = '#D55E00'.
    ``legend_cols`` : int, optional
        Number of columns in the legend. 
         Default = None.
    ``cols`` : list[str], optional
        List of colors for each MPC result. 
         Default = Category10_10.
    ``dash`` : list[str], optional
        List of line dash styles for each MPC result. 
         Default = ['solid', 'solid', 'solid', 'dashed', 'dotdash'].
    ``thickness`` : list[int], optional
        List of line widths for each MPC result. 
         Default = [8, 4, 4, 3, 3].
    ``alpha`` : list[float], optional
        List of line transparencies for each MPC result. 
         Default = [.75, .5, 1, 1, 1].
    ``latex_style`` : bool, optional
        Whether to apply LaTeX styling to the plots. 
         Default = False.

    Returns
    -------
    ``layout`` : bokeh.layouts.gridplot
        A vertical grid of Bokeh figures containing the MPC result plots.
    """
    # groups the results
    cMPC_groups: dict[str, list[AMPC_data | MPC_data]] = {key: list(group) for key, group in dataclass_group_by(MPC_results, group_by)}

    # styling attributes - in case there aren't enough, we will use default values
    c_cols = dict(zip(cMPC_groups.keys(), cols))
    l_dash = dict(zip(cMPC_groups.keys(), dash))
    l_thk = dict(zip(cMPC_groups.keys(), thickness))
    l_a = dict(zip(cMPC_groups.keys(), alpha))
    legend_it = defaultdict(list)
    
    # set of figures
    temp_MPC = MPC_results[0]
    P = temp_MPC.P
    nx, nu, x_rng, nt, dt = P.nx, P.nu, P.T_sim, P.N_sim, P.Ts
    xbnd = x_rng if xbnd is None else xbnd
    y_rng = [[-1.2 * P.xbnd[j,0], 1.2 * P.xbnd[j,0]] for j in range(nx)]
    
    if isinstance(theta_bnd, float):
        y_rng[1] = [-theta_bnd, theta_bnd] 
    elif isinstance(theta_bnd, Iterable):
        y_rng[1] = theta_bnd     

    
    activate_hover = activate_hover and len(MPC_results) < 20
    x_comb_range = Range1d(start=0, end=xbnd)
        
    ylabels_px = [r'$$x_{cart} \, [m]$$', r'$$\theta \, [\mathrm{rad}]$$', r'$$v \, [m/s]$$', r'$$\omega \, [\mathrm{rad}/s]$$'] 
    px = [figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_range=y_rng[i], y_axis_label=ylabels_px[i]) for i in range(nx)]
    pu = figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_axis_label=r'$$u \, [m/s^2]$$')
    pt = figure(background_fill_color=background_fill_color, width=width, height=height, x_range=x_comb_range, y_axis_label=r'$$t_{sol} \, [s]$$', y_axis_type=time_scale)
    pt.yaxis.ticker = AdaptiveTicker(desired_num_ticks=3)
    pt.yaxis.formatter = PrintfTickFormatter(format='%e')

    if additional_plots is not None and additional_plots:
        possible_add_plot_names = ['Iterations', 'Prep_Time', 'Fb_Time', 'Prep_Iterations', 'Fb_Iterations']
        if all(add_plot_name not in possible_add_plot_names for add_plot_name in additional_plots):
            raise AttributeError(f'Given \"additional_plots\" contain a name which is not an attribute of AMPC_data or MPC_data classes! \
                             \nValid options are: {possible_add_plot_names} \nYou provided: {additional_plots}')  
        
        if additional_plots_options is None:
            additional_plots_options = {}
        
        pi = {add_plot_name: figure(
            background_fill_color=background_fill_color, 
            width=width, 
            height=height, 
            x_range=x_comb_range, 
            **additional_plots_options.get(add_plot_name, {})
        ) for add_plot_name in additional_plots}
    

    is_first_plot = True
    for group_name, c_list in cMPC_groups.items(): 
        for cMPC in c_list:
            x = [i*dt for i in range(nt)]
            xi = [[(j+i)*dt for i in range(cMPC.X_traj.shape[2])] for j in range(nt)]
            Ts_ind = find_Ts(cMPC ,threshold) if plot_Ts else None

            legend_handles = []


            ## STATE PLOTS
            for state_idx in range(nx):
                temp_l_alpha = l_a.get(group_name,1)
                temp_l_width = l_thk.get(group_name,2)
                temp_l_dash = l_dash.get(group_name,'solid')
                temp_color = c_cols.get(group_name,'#D55E00')

                if is_first_plot:
                    # plot reference and state bounds
                    px[state_idx].line(x=x, y=0, line_width=2, color='black', line_dash= 'dotted')
                    bnd_top_x = BoxAnnotation(top=-P.xbnd[state_idx,0], fill_alpha=0.4, fill_color=bnd_color)
                    bnd_bot_x = BoxAnnotation(bottom=P.xbnd[state_idx,0], fill_alpha=0.4, fill_color=bnd_color)
                    px[state_idx].add_layout(bnd_top_x)
                    px[state_idx].add_layout(bnd_bot_x)
                    

                # open loop state trajectories
                if plot_mpc_trajectories:
                    state_traj_handle = px[state_idx].multi_line(
                        xs=xi,
                        ys=[cMPC.X_traj[i,state_idx,:] for i in range(nt)],
                        line_alpha=temp_l_alpha,
                        line_width=temp_l_width//2,
                        color=temp_color
                    )
                    legend_handles.append(state_traj_handle)
                
                # closed loop states plot
                state_handle = plot_line( 
                    px[state_idx], x, cMPC.X[state_idx,:],
                    line_alpha=temp_l_alpha,
                    line_width=temp_l_width,
                    line_dash=temp_l_dash,
                    color=temp_color,
                    hover_tool=activate_hover,
                )
                legend_handles.append(state_handle)
                
                # broken state cap point
                if np.isnan(cMPC.X[state_idx,-1]): 
                    last_pos = (~np.isnan(cMPC.X)).sum(axis = 1) - 1
                    broken_state_handle = px[state_idx].scatter(
                        x = x[last_pos[state_idx]],
                        y = cMPC.X[state_idx,last_pos[state_idx]],
                        size = 10, 
                        color = temp_color,
                        marker='star',
                    )
                    legend_handles.append(broken_state_handle)

                # settling point
                if Ts_ind is not None:
                    ts_handle = px[state_idx].scatter(
                        x = x[Ts_ind], 
                        y = cMPC.X[state_idx,Ts_ind],
                        size = 10,
                        color = temp_color
                    )
                    legend_handles.append(ts_handle)
            
            
            ## INPUT PLOT
            if is_first_plot:
                bnd_top_u = BoxAnnotation(top=-P.ubnd, fill_alpha=0.4, fill_color=bnd_color)
                bnd_bot_u = BoxAnnotation(bottom=P.ubnd, fill_alpha=0.4, fill_color=bnd_color)
                pu.add_layout(bnd_top_u)
                pu.add_layout(bnd_bot_u)

            # open loop input trajectories
            if plot_mpc_trajectories:
                ui = [[(j+i)*dt for i in range(cMPC.U_traj.shape[2])] for j in range(nt)]
                input_traj_handle = pu.multi_line(
                    xs=ui,
                    ys=[cMPC.U_traj[i,0,:] for i in range(nt)],
                    line_alpha=temp_l_alpha,
                    line_width=temp_l_width//2,
                    color=temp_color
                )
                legend_handles.append(input_traj_handle)
            
            # closed loop inputs plots
            input_handle = plot_line(
                pu, x, cMPC.U[0], 
                line_alpha=temp_l_alpha,
                line_width=temp_l_width,
                line_dash=temp_l_dash,
                color=temp_color,
                hover_tool=activate_hover,
            )
            legend_handles.append(input_handle)
            
            # broken input cap point
            if np.isnan(cMPC.U[0,-1]): 
                last_pos = (~np.isnan(cMPC.U)).sum(axis = 1) - 1
                broken_input_handle = pu.scatter(x=x[last_pos[0]], y=cMPC.U[0,last_pos[0]], size=10, color=temp_color, marker='star')
                legend_handles.append(broken_input_handle)


            # SOLVER TIMES
            soltime_handle = plot_line(
                pt, x, cMPC.Acados_Time if isinstance(cMPC, AMPC_data) and time_type == 'acados' else cMPC.Time,
                line_alpha=temp_l_alpha,
                line_width=temp_l_width,
                line_dash=temp_l_dash,
                color=temp_color,
                hover_tool=activate_hover,
            )
            legend_handles.append(soltime_handle)


            # ADDITIONAL PLOTS
            if additional_plots is not None:
                for add_plot_name in additional_plots:
                    if hasattr(cMPC, add_plot_name):
                        add_plot_handle = plot_line(
                            pi[add_plot_name], x, getattr(cMPC, add_plot_name),
                            line_alpha=temp_l_alpha,
                            line_width=temp_l_width,
                            line_dash=temp_l_dash,
                            color=temp_color,
                            hover_tool=activate_hover,
                        )
                        legend_handles.append(add_plot_handle)

            legend_it[group_name].extend(legend_handles)

            is_first_plot = False
    

    # LEGEND
    legend_it = list(legend_it.items())
    legend = Legend(items=legend_it, orientation="horizontal", click_policy="hide", title=legend_title)
    legend.ncols = calculate_ncols(width, legend) if legend_cols is None else legend_cols
    nrows = (len(legend.items) + legend.ncols - 1) // legend.ncols
    estim_legend_height = legend.spacing * (nrows - 1) + max(legend.label_height, legend.glyph_height) * nrows + 2 * legend.padding + 2 * legend.margin # get_legend_height(legend)

    p_add = [*px, pu, pt] if time_type is not None else [*px, pu]
    if additional_plots is not None:
        p_add.extend([pi[add_plot_name] for add_plot_name in additional_plots])

    # change the height of last figure and add legend
    p_add[-1].xaxis.axis_label = r'$$t_{sim}$$'
    p_add[-1].height += estim_legend_height + p_add[-1].xaxis.axis_label_standoff + pt2px_height(int(p_add[-1].xaxis.axis_label_text_font_size[:-2]))
    p_add[-1].add_layout(legend, 'below')
    

    for ii in range(len(p_add)-1):
        p_add[ii].xaxis.major_tick_line_color = None
        p_add[ii].xaxis.minor_tick_line_color = None
        p_add[ii].xaxis.major_label_text_font_size = '0pt'


    # SELECTION BUTTON
    select = RadioButtonGroup(labels=["Show all", "Hide all"], active=0)
    all_renderers = list(chain.from_iterable((r[1] for r in legend_it)))
    callback = CustomJS(args=dict(renderers=all_renderers), code="""
        var visibility = cb_obj.active == 0;
        for (var i = 0; i < renderers.length; i++) {
            renderers[i].visible = visibility;
        }
    """)
    select.js_on_change('active', callback)
    p = [select] + p_add


    layout = gridplot([[x] for x in p])
    if latex_style:
        set_figure_to_default_latex(layout, move_legend_top=False, output_backend='svg')

    return layout



def find_Ts(MPC_result: AMPC_data | MPC_data, threshold=None) -> int | None:
    """
    Finds the time step index corresponding to the settling time of the closed-loop simulation results.

    Parameters
    ----------
    ``MPC_result`` : AMPC_data | MPC_data
        The MPC result object containing simulation data. This should include:
        - `X`: a 2D numpy array of predicted states.
        - `P`: an instance of the MPC_Param dataclass with problem parameters.
    ``threshold`` : float, optional
        The threshold value used to determine if the control input is considered close enough to the reference. If None, a default threshold of 5% of the state bounds is used. Default = None.

    Returns
    -------
    ``Ts_ind`` :  int | None
        The index of the last time step where the state is within the threshold, 
         or None if the state does not settle within the simulation time.
    """
    X,P = MPC_result.X, MPC_result.P
    if threshold is None:
        # use 5% of the xbnd as default
        threshold = P.xbnd*0.05

    Ts_ind = np.where(np.invert((np.abs(X)<threshold).all(axis=0)))[0][-1]
    if Ts_ind < X.shape[1]-1:
        return Ts_ind
    else:
        return None



def heatmap(
        df: pd.DataFrame, 
        x_label: str,
        y_label: str,
        cbar_label: str,
        # base_series: Optional[pd.Series] = None,
        color_palette = Turbo256, 
        cmap_cap: float = np.nan,
        cbar_unit: str = None,
        title = '',
        latex_style: bool = False,
        figure_size: tuple[float] = (800, 800),
    ):
    """
    Creates a heatmap with the DataFrame columns as x-axis and the DataFrame index as y-axis. 
    A colorbar is included to represent the values, with an optional cap on the maximum value.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        The source DataFrame for the heatmap, with columns for x-axis, y-axis, and color values.
    ``x_label`` : str
        The name of the column used for the x-axis.
    ``y_label`` : str
        The name of the column used for the y-axis.
    ``cbar_label`` : str
        The name of the column used for the color values.
    ``color_palette`` : colormap, optional
        The colormap instance used for the heatmap colors. Default = Turbo256.
    ``cmap_cap`` : float, optional
        The maximum value for the colorbar. Default = np.nan.
    ``cbar_unit`` : str, optional
        The unit of the displayed values on the colorbar, e.g., 'ms'. Default = None.
    ``title`` : str, optional
        The title of the heatmap figure. Default = ''.
    ``latex_style`` : bool, optional
        Whether to apply LaTeX style formatting to the figure. Default = False.
    ``figure_size`` : tuple of float, optional
        The width and height of the figure in pixels. Default = (800, 800).

    Returns
    -------
    ``figure`` : bokeh.plotting.figure
        A Bokeh figure object containing the heatmap with a colorbar.
    """
    df = df.copy()

    x_range = np.sort(df[x_label].unique()).astype(str)
    y_range = np.sort(df[y_label].unique())[::-1].astype(str)

    df[x_label] = df[x_label].astype(str)
    df[y_label] = df[y_label].astype(str)
    
    # Creating color maps for the upper left and lower right triangles
    low = df[cbar_label].values.min()
    high = min(df[cbar_label].values.max(), cmap_cap)
    cmap_soltimes = linear_cmap(
        field_name=cbar_label, 
        palette=color_palette, 
        low=low, 
        high=high
    )

    source = ColumnDataSource(df)

    # Setting up the Bokeh figure
    p = figure(
        width = figure_size[0], 
        height = figure_size[1],  
        title = title,
        x_range = x_range,
        y_range = y_range,
        x_axis_label = latex_labels[x_label],
        y_axis_label = latex_labels[y_label],
        x_axis_location = "above",
        toolbar_location = 'below',
        tools = ['pan']
    )

    # hover tool
    hover = HoverTool(tooltips = [(cbar_label, f'@{cbar_label}' if cbar_unit is None else f'@{cbar_label} {cbar_unit}')])
    p.add_tools(hover)

    # rectangles
    r = p.rect(
        x=x_label, 
        y=y_label, 
        width=1, 
        height=1, 
        source=source,
        fill_color=cmap_soltimes,
        line_color='white',
    )

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "12pt"
    p.axis.major_label_standoff = 0
    p.axis.axis_label_text_font_size = '14pt'
    p.title.text_font_size = '16pt'


    # Adding color bars for each triangle
    # ticks = np.linspace(low, high, num=5) + base_series[cbar_label]
    # ticker = FixedTicker(ticks=ticks)
    cbar = r.construct_color_bar(
        major_label_text_font_size = '12pt',
        label_standoff = 6,
        # ticker=ticker,
        border_line_color = 'white',
        title = latex_label_with_unit(cbar_label, cbar_unit),
        title_standoff = 5,
        title_text_font_size = '16pt',
        title_text_align = 'right',
    )
    p.add_layout(cbar, 'right')

    if latex_style:
        set_figure_to_default_latex(p, output_backend='svg')

    return p




def _plot_box(
        p: figure, 
        df: pd.DataFrame, 
        value_name: str, 
        category_name: str, 
        box_color: Iterable,
        scatter_color: Iterable, 
        show_outliers: bool = False,
        show_non_outliers: bool = False,
    ):
    """
    Plots a box plot on the provided Bokeh figure, including whiskers, quantile boxes, 
    and optional scatter points for outliers and non-outliers.

    Parameters
    ----------
    ``p`` : bokeh.plotting.figure
        The Bokeh figure object to plot on.
    ``df`` : pd.DataFrame
        The DataFrame containing the data to be plotted. Must include columns for the category, 
        value, and optional indicators for outliers.
    ``value_name`` : str
        The name of the column in `df` that contains the values to be plotted.
    ``category_name`` : str
        The name of the column in `df` that contains the categories for the box plot.
    ``box_color`` : iterable
        Color or sequence of colors for the box plot elements.
    ``scatter_color`` : iterable
        Color or sequence of colors for the scatter points.
    ``show_outliers`` : bool, optional
        Whether to display scatter points for outliers. Default is False.
    ``show_non_outliers`` : bool, optional
        Whether to display scatter points for non-outliers. Default is False.

    Returns
    -------
    ``whisker_renderers``: list
        List of Bokeh renderers for the whiskers.
    ``box_renderers``: list
        List of Bokeh renderers for the quantile boxes.
    ``scatter_renderers``: list
        List of Bokeh renderers for the scatter points (if any are displayed).
    """
    boxes_source = ColumnDataSource(df)

    whisker_renderers = []
    box_renderers = []
    scatter_renderers = []

    # whiskers
    whisker = Whisker(base=category_name, upper='upper', lower='lower', source=boxes_source, line_alpha=0.8)
    whisker.upper_head.size = whisker.lower_head.size = 20
    whisker_renderers.append(whisker)
    p.add_layout(whisker)

    # quantile boxes
    # box_cmap = factor_cmap(category_name, box_colors, categories)
    quantbox_top = p.vbar(category_name, 0.7, 'q2', 'q3', source=boxes_source, fill_color=box_color, line_color='black')
    quantbox_bot = p.vbar(category_name, 0.7, 'q1', 'q2', source=boxes_source, fill_color=box_color, line_color='black')
    box_renderers.append(quantbox_top)
    box_renderers.append(quantbox_bot)
    
    # scatters
    # scatter_cmap = factor_cmap(category_name, scatter_colors, categories)
    if show_outliers:
        outliers_source = ColumnDataSource(df[df['is_outlier']])
        outl_r = p.scatter(jitter(category_name, width=0.35, range=p.x_range), value_name, source=outliers_source, size=6, color=scatter_color, alpha=0.8)
        scatter_renderers.append(outl_r)

    if show_non_outliers:
        non_outliers_source = ColumnDataSource(df[df['is_non_outlier']])
        non_outl_r = p.scatter(jitter(category_name, width=0.35, range=p.x_range), value_name, source=non_outliers_source, size=6, color=scatter_color, alpha=0.8)
        scatter_renderers.append(non_outl_r)

    return whisker_renderers, box_renderers, scatter_renderers



def boxplot(
        df: pd.DataFrame, 
        category_names: str | list[str],
        value_name: str,
        category_sort_fun: Optional[Callable] = None,
        y_unit: Optional[str] = None,
        x_unit: Optional[str] = None,
        show_outliers: bool = False,
        show_non_outliers: bool = False,
        legend_category: Optional[int | str] = None,
        box_colors: list[str] = Viridis10, 
        scatter_colors: list[str]= Viridis10, 
        title: str = '',
        figure_size: tuple[float] = (800, 800),
        y_range: tuple = None,
        latex_style: bool = False,
        hover_tooltips: list[str] = None,
    ):
    """
    Creates a boxplot with optional scatter plots for visualizing data distribution.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        The DataFrame containing the data to plot.
    ``category_names`` : str | list[str]
        The name(s) of the column(s) used for categorizing the data.
    ``value_name`` : str
        The name of the column containing the values to plot (y-axis).
    ``category_sort_fun`` : Optional[Callable], default=None
        A function to sort categories, applied if multiple categories are provided.
    ``y_unit`` : Optional[str], default=None
        The unit for the y-axis labels.
    ``x_unit`` : Optional[str], default=None
        The unit for the x-axis labels.
    ``show_outliers`` : bool, default=False
        Whether to display outliers.
    ``show_non_outliers`` : bool, default=False
        Whether to display non-outliers.
    ``legend_category`` : Optional[int | str], default=None
        The category to use for the legend.
    ``box_colors`` : list[str], default=Viridis10
        Colors for the boxplots.
    ``scatter_colors`` : list[str], default=Viridis10
        Colors for the scatter plots.
    ``title`` : str, default=''
        The title of the plot.
    ``figure_size`` : tuple[float], default=(800, 800)
        The size of the figure (width, height).
    ``y_range`` : Optional[tuple], default=None
        The range for the y-axis.
    ``latex_style`` : bool, default=False
        Whether to use LaTeX style for labels.
    ``hover_tooltips`` : Optional[list[str]], default=None
        Additional tooltips for hover information.

    Returns
    -------
    ``p`` : figure
        The Bokeh figure object containing the boxplot.
    """
    df = df.copy()
    combi_cat_name = 'Combined_Categories'
    legend_name = 'Legend'

    if type(category_names) == list and len(category_names) > 1:
        if category_sort_fun is None:
            category_sort_fun = lambda x: '_'.join(x)

        df[category_names] = df[category_names].astype(str)
        df[combi_cat_name] = df[category_names].apply(tuple, axis=1)
        df[legend_name] = df[category_names[1:]].apply('_'.join, axis=1)
        sorted_categories = sorted(df[combi_cat_name].unique(), key=category_sort_fun)
        categories = [tuple(str(cat_val) for cat_val in cat_sorted) for cat_sorted in sorted_categories]    
    else:
        if category_sort_fun is None:
            category_sort_fun = lambda x: int(x)

        df[combi_cat_name] = df[category_names].astype(str)
        sorted_categories = sorted(df[combi_cat_name].unique(), key=category_sort_fun)
        categories = [str(cat) for cat in sorted_categories]
    

    # compute quantiles
    qs = df.groupby(combi_cat_name)[value_name].quantile([0.25, 0.5, 0.75])
    qs = qs.unstack().reset_index()
    qs.columns = [combi_cat_name, 'q1', 'q2', 'q3']
    df = pd.merge(df, qs, on=combi_cat_name, how='left')

    # compute IQR outlier bounds
    iqr = df['q3'] - df['q1']
    df['upper'] = df['q3'] + 1.5*iqr
    df['lower'] = df['q1'] - 1.5*iqr

    # find outliers and non outliers
    df['is_non_outlier'] = (df[value_name] >= df['lower']) & (df[value_name] <= df['upper'])
    df['is_outlier'] = (df[value_name] < df['lower']) | (df[value_name] > df['upper'])

    df.columns = df.columns.astype(str)
    if not isinstance(df.dtypes[combi_cat_name], str | object):
        raise TypeError(f'Wrong type of category column {combi_cat_name} -> {df.dtypes[combi_cat_name]}')

    # Setting up the Bokeh figure
    p = figure(
        width = figure_size[0], 
        height = figure_size[1], 
        title = title,
        x_range=FactorRange(*categories),
        y_axis_label = latex_label_with_unit(value_name, y_unit),
        x_axis_label = latex_label_with_unit(category_names[0] if type(category_names) == list else category_names, x_unit),
        toolbar_location = 'below',
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset']
    )
    if y_range is not None:
        p.y_range = Range1d(start=y_range[0], end=y_range[1])


    # Plot scatters and boxes and whiskerers
    if legend_category is None and type(category_names) == str:
        group_name = category_names
    elif legend_category is None and type(category_names) == list:
        group_name = category_names[0]
    elif legend_category is not None and type(legend_category) == int:
        group_name = category_names[legend_category]
    elif legend_category is not None and type(legend_category) == str and type(category_names) == list and legend_category in category_names:
        group_name = legend_category
    else:
        raise ValueError(f'"{legend_category}" need to be a string or integer! Please probide an appropriate "color_group_index".')

    whisker_renderers = defaultdict(list)
    box_renderers = defaultdict(list)
    scatter_renderers = defaultdict(list)
    for i, (cat_name, grouped_df) in enumerate(df.groupby(group_name)):
        w_rend, b_rend, s_rend = _plot_box(p, grouped_df, value_name, combi_cat_name, box_colors[i], scatter_colors[i], show_outliers, show_non_outliers)
        whisker_renderers[cat_name].extend(w_rend)
        box_renderers[cat_name].extend(b_rend)
        scatter_renderers[cat_name].extend(s_rend)
            
    # Hover tool
    tooltips = [
        (value_name, f'@{value_name}' if y_unit is None else f'@{value_name} {y_unit}'),
    ]
    if hover_tooltips is not None:
        tooltips = [*tooltips, *[(hover_tooltip, f'@{hover_tooltip}') for hover_tooltip in hover_tooltips]]

    hover_renderers = [rend for rend_list in scatter_renderers.values() for rend in rend_list]
    hover = HoverTool(
        tooltips = tooltips,
        renderers = hover_renderers
    )
    p.add_tools(hover)

    # plot settings
    p.grid.grid_line_color = None
    p.yaxis.major_label_text_font_size = '12pt'

    if type(category_names) == list and len(category_names) > 1:
        p.xaxis.major_label_text_font_size = '0pt'
        p.xaxis.group_text_font_size = '12pt'
        p.xaxis.group_text_color = 'black'
        p.xaxis.group_text_font_style = 'normal'

        if legend_category is not None:
            legend_it = merge_listdicts(box_renderers, scatter_renderers)
            legend_it = list(legend_it.items())
            legend = Legend(items=legend_it, orientation='horizontal', click_policy='hide')
            legend.ncols = calculate_ncols(p.width, legend)
            p.height += get_legend_height(legend)
            p.add_layout(legend, 'below')

    else:
        p.axis.major_label_text_font_size = '12pt'
        # p.xaxis.major_label_orientation = pi*0.5

    p.axis.axis_label_text_font_size = '14pt'
    p.title.text_font_size = '16pt'

    if latex_style:
        set_figure_to_default_latex(p, line_width=1, output_backend='svg')

    return p



def line_with_scatter(
        df: pd.DataFrame, 
        x_label: str,
        y_label: str,
        line_name: str,
        show_points = False,
        show_means = False,
        highlight_max = False,
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        line_unit: Optional[str] = None,
        title = '',
        colors = ('#0072B2', '#E69F00', '#D55E00','#333333', '#009E73'), 
        dash = ('solid','solid','solid','dashed','dotdash'),
        figure_size: tuple[float] = (1200, 800),
        latex_style: bool = False,
    ):
    """
    Plots a line chart with optional scatter points and means.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        The DataFrame containing the data to plot.
    ``x_label`` : str
        The name of the column for the x-axis.
    ``y_label`` : str
        The name of the column for the y-axis.
    ``line_name`` : str
        The name of the column for the line grouping.
    ``show_points`` : bool, default=False
        Whether to display individual data points.
    ``show_means`` : bool, default=False
        Whether to display the mean values.
    ``highlight_max`` : bool, default=False
        Whether to highlight the maximum mean values.
    ``x_unit`` : Optional[str], default=None
        The unit for the x-axis labels.
    ``y_unit`` : Optional[str], default=None
        The unit for the y-axis labels.
    ``line_unit`` : Optional[str], default=None
        The unit for the line labels in the legend.
    ``title`` : str, default=''
        The title of the plot.
    ``colors`` : tuple[str], default=('#0072B2', '#E69F00', '#D55E00','#333333', '#009E73')
        Colors for the lines and scatter points.
    ``dash`` : tuple[str], default=('solid','solid','solid','dashed','dotdash')
        Dash styles for the lines.
    ``figure_size`` : tuple[float], default=(1200, 800)
        The size of the figure (width, height).
    ``latex_style`` : bool, default=False
        Whether to use LaTeX style for labels.

    Returns
    -------
    ``p`` : figure
        The Bokeh figure object containing the line plot with scatter points.
    """
    df = df.copy()

    mean_suff = '_mean'
    mean_name = y_label + mean_suff

    df[line_name] = df[line_name].astype(str)
    lines = df[line_name].unique()
    num_lines = len(lines)

    # specify colors
    if len(colors) < num_lines:
        colors = Category20[num_lines]
    cols = {k: v for k, v in zip(lines, colors)}
    dashes = {k: v for k, v in zip(lines, dash)}

    # make means
    df_means = df.groupby([line_name, x_label]).mean().add_suffix(mean_suff).reset_index()

    # calc maximums of the means of one line
    max_R2_per_group = df_means.groupby(line_name)[mean_name].transform('max')
    df_max = df_means[df_means[mean_name] == max_R2_per_group].reset_index(drop=True)

    # figure
    p = figure(
        width = figure_size[0], 
        height = figure_size[1],
        title = title,
        y_axis_label = latex_label_with_unit(y_label, y_unit),
        x_axis_label = latex_label_with_unit(x_label, x_unit),
        toolbar_location = 'below',
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset']
    )

    # lines and scatters
    legend_it = defaultdict(list)
    hover_renderers_means = []
    hover_renderers_points = []
    for (name, group_df_means), (name, group_df_points), (name, group_df_max) in zip(
        df_means.groupby(line_name), 
        df.groupby(line_name),
        df_max.groupby(line_name),
    ):
        source_means = ColumnDataSource(group_df_means)

        r_line = p.line(
            x = x_label, 
            y = mean_name, 
            source = source_means, 
            line_width = 4, 
            line_color = cols[name], 
            line_dash = dashes[name],
        )
        legend_it[name].append(r_line)

        if show_points:
            source_points = ColumnDataSource(group_df_points)

            r_scatter_points = p.scatter(
                x = x_label, 
                y = y_label, 
                source = source_points, 
                size = 6, 
                alpha = 0.6, 
                fill_color = cols[name],
                line_color='black',
            )
            hover_renderers_points.append(r_scatter_points)
            legend_it[name].append(r_scatter_points)

        if show_means:
            r_scatter_means = p.scatter(
                x = x_label, 
                y = mean_name, 
                source = source_means, 
                size = 8, 
                alpha = 1., 
                line_width = 3,
                fill_color = 'white',
                line_color = cols[name],
            )
            hover_renderers_means.append(r_scatter_means)
            legend_it[name].append(r_scatter_means)

        if highlight_max:
            source_max = ColumnDataSource(group_df_max)

            r_scatter_max = p.scatter(
                x = x_label, 
                y = mean_name, 
                source = source_max, 
                size = 16, 
                alpha = 1., 
                marker = 'star',
                fill_color = 'red',
                line_color = 'red',
            )
            hover_renderers_means.append(r_scatter_max)
            legend_it[name].append(r_scatter_max)


    if show_means:
        hover_means = HoverTool(
            tooltips = [(mean_name, f'@{mean_name}'), (x_label, f'@{x_label}')],
            renderers = hover_renderers_means,
        )
        p.add_tools(hover_means)

    if show_points:
        hover_means = HoverTool(
            tooltips = [(y_label, f'@{y_label}')],
            renderers = hover_renderers_points
        )
        p.add_tools(hover_means)
        

    # # Color bar
    # color_mapper = CategoricalColorMapper(factors=list(cols.keys()), palette=list(cols.values()))
    # cbar = ColorBar(
    #     color_mapper=color_mapper,
    #     major_label_text_font_size = '12pt',
    #     label_standoff = 6,
    #     border_line_color = 'white',
    #     title = latex_label_with_unit(line_name, line_unit),
    #     title_standoff = 5,
    #     title_text_font_size = '16pt',
    #     title_text_align = 'right',
    # )
    # p.add_layout(cbar, 'right')

    p.grid.grid_line_color = None
    p.axis.major_label_text_font_size = '12pt'
    p.axis.axis_label_text_font_size = '14pt'
    p.title.text_font_size = '16pt'

    legend_it = list(legend_it.items())
    legend = Legend(items=legend_it, orientation='horizontal', click_policy='hide', title=latex_label_with_unit(line_name, line_unit))
    legend.ncols = calculate_ncols(p.width, legend)
    p.height += get_legend_height(legend)
    p.add_layout(legend, 'below')

    if latex_style:
        set_figure_to_default_latex(p, output_backend='svg')

    return p




def scatter(
        df: pd.DataFrame,
        y_label: str,
        x_label: str,
        cbar_label: str,
        legend_label: str,
        title: str = '',
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        cbar_unit: Optional[str] = None,
        legend_unit: Optional[str] = None,
        baseline_df: Optional[pd.Series] = None, 
        meanmedian_df: Optional[pd.DataFrame] = None,
        x_range: Optional[tuple[float, float]] = None,
        y_range: Optional[tuple[float, float]] = None,
        figure_size: tuple[float] = (1200, 800),
        latex_style: bool = False,
        color_palette = Category10_10,
        markers = ['asterisk', 'circle', 'triangle', 'square', 'star', 'hex', 'plus', 'x', 'y'],
):
    """
    Creates a scatter plot with optional baseline and mean/median overlays, and interactive legends.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        DataFrame containing the data to be plotted.
    ``y_label`` : str
        Label for the y-axis.
    ``x_label`` : str
        Label for the x-axis.
    ``cbar_label`` : str
        Label for the color bar.
    ``legend_label`` : str
        Label for the legend.
    ``title`` : str, optional
        Title of the plot.
         Default = ''.
    ``x_unit`` : str, optional
        Unit for the x-axis label.
    ``y_unit`` : str, optional
        Unit for the y-axis label.
    ``cbar_unit`` : str, optional
        Unit for the color bar label.
    ``legend_unit`` : str, optional
        Unit for the legend label.
    ``baseline_df`` : pd.Series, optional
        Series to plot baseline data.
    ``meanmedian_df`` : pd.DataFrame, optional
        DataFrame containing mean or median values to overlay.
    ``x_range`` : tuple[float, float], optional
        Range for the x-axis.
    ``y_range`` : tuple[float, float], optional
        Range for the y-axis.
    ``figure_size`` : tuple[float], optional
        Size of the figure in pixels.
         Default = (1200, 800).
    ``latex_style`` : bool, optional
        Whether to apply LaTeX style to the plot.
         Default = False.
    ``color_palette`` : iterable, optional
        Color palette for the scatter plot.
         Default = Category10_10.
    ``markers`` : list, optional
        List of markers to use in the scatter plot.
         Default = ['asterisk', 'circle', 'triangle', 'square', 'star', 'hex', 'plus', 'x', 'y'].

    Returns
    -------
    ``p`` : bokeh.plotting.figure.Figure
        The Bokeh figure object containing the scatter plot.
    """
    df = df.copy()
    df[cbar_label] = df[cbar_label].astype(str)
    df[legend_label] = df[legend_label].astype(str)

    p = figure(
        width = figure_size[0], 
        height = figure_size[1],
        title = title,
        y_axis_label = latex_label_with_unit(y_label, y_unit),
        x_axis_label = latex_label_with_unit(x_label, x_unit),
        toolbar_location = 'below',
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset']
    )

    # add limits to plot
    if x_range is not None:
        p.x_range = Range1d(start=x_range[0], end=x_range[1])
    if y_range is not None:
        p.y_range = Range1d(start=y_range[0], end=y_range[1])


    # plot baseline mpc
    if baseline_df is not None and x_label in baseline_df.index:
        x_span = p.add_layout(Span(location=baseline_df[x_label], dimension='height', line_color='black', line_width=3, line_dash='dashed'))
    if baseline_df is not None and y_label in baseline_df.index:
        y_span = p.add_layout(Span(location=baseline_df[y_label], dimension='width', line_color='black', line_width=3, line_dash='dashed'))


    ## =================================== UNPROCESSED PLOT ====================================
    legend_it_unpr = defaultdict(list)
    renderers_unpr = []

    # create color and mark maps for normal scatter
    cmap_factors = sorted(df[cbar_label].unique(), key=lambda x: float(x))
    cmap = factor_cmap(factors=cmap_factors, field_name=cbar_label, palette=color_palette[:len(cmap_factors)])

    marks_factors = sorted(df[legend_label].unique(), key=lambda x: float(x))
    mark_map = factor_mark(factors=marks_factors, field_name=legend_label, markers=markers[:len(marks_factors)])

    # grouped scatter plot
    for name, df_group in df.groupby(legend_label):
        grouped_source = ColumnDataSource(df_group)
        r = p.scatter(
            x = x_label, 
            y = y_label, 
            source = grouped_source, 
            size = 6, 
            alpha = 0.3 if meanmedian_df is not None else 0.9, 
            color = cmap,
            marker = mark_map,
        )
        legend_it_unpr[name].append(r)
        renderers_unpr.append(r)

    ## =================================== MEAN/MEDIAN PLOT ====================================
    legend_it_mm = defaultdict(list)
    renderers_mm = []
    if meanmedian_df is not None:
        meanmedian_df[cbar_label] = meanmedian_df[cbar_label].astype(str)
        meanmedian_df[legend_label] = meanmedian_df[legend_label].astype(str)

        # create color and mark maps for meadian and mean scatter
        mm_cmap_factors = sorted(meanmedian_df[cbar_label].unique(), key=lambda x: float(x))
        mm_cmap = factor_cmap(factors=mm_cmap_factors, field_name=cbar_label, palette=color_palette[:len(mm_cmap_factors)])

        mm_marks_factors = sorted(meanmedian_df[legend_label].unique(), key=lambda x: float(x))
        mm_mark_map = factor_mark(factors=mm_marks_factors, field_name=legend_label, markers=markers[:len(mm_marks_factors)])

        # grouped scatter plot
        for name, mm_df_group in meanmedian_df.groupby(legend_label):
            mm_grouped_source = ColumnDataSource(mm_df_group)
            r = p.scatter(
                x = x_label, 
                y = y_label, 
                source = mm_grouped_source, 
                size = 9, 
                alpha = 1, 
                color = mm_cmap,
                marker = mm_mark_map,
            )
            legend_it_mm[name].append(r)
            renderers_mm.append(r)


    # Hovertool
    hover_means = HoverTool(
        tooltips = [
            (y_label, f'@{y_label}'),
            (x_label, f'@{x_label}'),
            (cbar_label, f'@{cbar_label}'),
            (legend_label, f'@{legend_label}')
        ],
        renderers = renderers_unpr + renderers_mm,
    )
    p.add_tools(hover_means)

    # Colorbar
    cbar = r.construct_color_bar(
        major_label_text_font_size = '12pt',
        label_standoff = 6,
        border_line_color = 'white',
        title_standoff = 5,
        title_text_font_size = '16pt',
        title_text_align = 'right',
        title = latex_label_with_unit(cbar_label, cbar_unit),
    )
    p.add_layout(cbar, 'right')


    ## ================================ LEGENDS =====================================
    # Standard Legend
    legend_unpr = Legend(
        items=sorted(legend_it_unpr.items(), key=lambda x: int(x[0])), 
        orientation='horizontal', click_policy='hide', title=latex_label_with_unit(legend_label, legend_unit)
    )
    legend_unpr.ncols = calculate_ncols(p.width, legend_unpr)
    p.add_layout(legend_unpr, 'below')

    if meanmedian_df is not None:
        # Mean/Median Legend
        legend_mm = Legend(
            items=sorted(legend_it_mm.items(), key=lambda x: int(x[0])), 
            orientation='horizontal', click_policy='hide', title=latex_label_with_unit(legend_label, legend_unit)
        )
        legend_mm.ncols = calculate_ncols(p.width, legend_mm)
        p.add_layout(legend_mm, 'below')

        # Combined Legend
        legend_it_combi = {key: legend_it_unpr.get(key, []) + legend_it_mm.get(key, []) for key in set(legend_it_unpr) | set(legend_it_mm)}
        legend_combi = Legend(
            items=sorted(legend_it_combi.items(), key=lambda x: int(x[0])), 
            orientation='horizontal', click_policy='hide', title=latex_label_with_unit(legend_label, legend_unit)
        )
        legend_combi.ncols = calculate_ncols(p.width, legend_combi)
        p.add_layout(legend_combi, 'below')
    else:
        legend_mm = Legend()
        legend_combi = Legend()

    p.height += get_legend_height(legend_unpr)


    # other settings
    p.axis.major_label_text_font_size = '12pt'
    p.axis.axis_label_text_font_size = '14pt'
    p.title.text_font_size = '16pt'

    # ============================================ SELECT =============================
    # Radio select Hide or Show all
    callback_radio = CustomJS(args=dict(unpr_rend=renderers_unpr, mm_rend=renderers_mm, legend_unpr=legend_unpr, legend_mm=legend_mm, legend_combi=legend_combi), code="""
            var selected_button = cb_obj.active;
                            
            var unpr_visible = false;
            var mm_visible = false;

            if (selected_button == 0) {
                if (legend_combi.visible) {
                    unpr_visible = true;
                    mm_visible = true;
                } else if (legend_unpr.visible) {
                    unpr_visible = true;
                } else if (legend_mm.visible) {
                    mm_visible = true;
                }
            }

            for (var i = 0; i < unpr_rend.length; i++) {
                unpr_rend[i].visible = unpr_visible;
            }
            for (var i = 0; i < mm_rend.length; i++) {
                mm_rend[i].visible = mm_visible;
            }
        """)
    radio_button_group = RadioButtonGroup(labels=['Show all', 'Hide all'], active=0)
    radio_button_group.js_on_change('active', callback_radio)

    if meanmedian_df is not None:
        # add checkbox for showing scatters
        legend_unpr.visible = False
        legend_mm.visible = False
        legend_combi.visible = True
        callback_legend = CustomJS(args=dict(unpr_rend=renderers_unpr, mm_rend=renderers_mm, legend_unpr=legend_unpr, legend_mm=legend_mm, legend_combi=legend_combi), code="""
            var selected_option = cb_obj.value;
            var unpr_visible = false;
            var mm_visible = false;
            var unpr_alpha = 0.0;
            var mm_alpha = 0.0;
                            
            if (selected_option == 'All') {
                legend_combi.visible = true
                legend_unpr.visible = false;
                legend_mm.visible = false;
                unpr_visible = true;
                mm_visible = true;
                unpr_alpha = 0.3;
                mm_alpha = 1.0;
                            
            } else if (selected_option == 'Means/Medians') {
                legend_combi.visible = false
                legend_unpr.visible = false;
                legend_mm.visible = true;
                                   
                unpr_visible = false;
                mm_visible = true;
                unpr_alpha = 0.0;
                mm_alpha = 1.0;       
                            
            } else if (selected_option == 'Unprocessed') {
                legend_combi.visible = false
                legend_unpr.visible = true;
                legend_mm.visible = false;
                                   
                unpr_visible = true;
                mm_visible = false;
                unpr_alpha = 1.0;
                mm_alpha = 0.0;       
            }
                                   
            for (var i = 0; i < unpr_rend.length; i++) {
                unpr_rend[i].visible = unpr_visible;
                unpr_rend[i].glyph.fill_alpha = unpr_alpha;
                unpr_rend[i].glyph.line_alpha = unpr_alpha;
            }
            for (var i = 0; i < mm_rend.length; i++) {
                mm_rend[i].visible = mm_visible;
                mm_rend[i].glyph.fill_alpha = mm_alpha;
                mm_rend[i].glyph.line_alpha = mm_alpha;
            }
        """)
        select = Select(title='What to show ', options=['All', 'Means/Medians', 'Unprocessed'], value='All')
        select.js_on_change('value', callback_legend)
        p = column(select, radio_button_group, p)
    else:
        legend_unpr.visible = True
        legend_mm.visible = False
        legend_combi.visible = False
        p = column(radio_button_group, p)

    if latex_style:
        set_figure_to_default_latex(p, output_backend='svg')

    return p



def histogram_pdf(
        df: pd.DataFrame, 
        x_label: str,
        group_by_values: list | str,
        color_palette = Category10_10, 
        title: str = '',
        figure_size: tuple[float] = (800, 800),
        bins: Optional[list | int] = None,
        x_range: Optional[tuple[float, float]] = None,
        legend_label_callable: Optional[Callable[[tuple], str]] = None,
        legend_cols: Optional[int] = None,
        cap_value: Optional[float] = None,
        latex_style: bool = False,
):
    """
    Creates a histogram and probability density function (PDF) plot for a given dataset 
    with options for customization.

    Parameters
    ----------
    ``df`` : pd.DataFrame
        DataFrame containing the data to be plotted.
    ``x_label`` : str
        Label for the x-axis.
    ``group_by_values`` : list | str
        Column name or list of column names to group the data by.
    ``color_palette`` : iterable, optional
        Color palette for the plot.
         Default = Category10_10.
    ``title`` : str, optional
        Title of the plot.
         Default = ''.
    ``figure_size`` : tuple[float], optional
        Size of the figure in pixels.
         Default = (800, 800).
    ``bins`` : list | int, optional
        Number of bins or a list of bin edges for the histogram.
    ``x_range`` : tuple[float, float], optional
        Range for the x-axis.
    ``legend_label_callable`` : Callable[[tuple], str], optional
        Function to generate legend labels from group names.
    ``legend_cols`` : int, optional
        Number of columns in the legend.
    ``cap_value`` : float, optional
        Maximum value for the x-axis data.
    ``latex_style`` : bool, optional
        Whether to apply LaTeX style to the plot.
         Default = False.

    Returns
    -------
    ``p`` : bokeh.plotting.figure.Figure
        The Bokeh figure object containing the histogram and PDF plot.
    """
    df = df.copy()

    if cap_value is not None:
        df.loc[df[x_label] > cap_value, x_label] = cap_value

    x_hist = df[x_label]
    x_min = x_hist.min()
    x_max = x_hist.max()

    if bins is None:
        bins = 20
    if type(bins) == int:
        bins = np.linspace(x_min, x_max, bins)
    
    if legend_label_callable is None:
        legend_label_callable = lambda x: ', '.join(str(value) for value in x)

    p = figure(
        width = figure_size[0], 
        height = figure_size[1],
        title = title,
        y_axis_label = r'$$\mathrm{PDF}(x)$$',
        x_axis_label = x_label,
        toolbar_location = 'below',
        tools = ['pan', 'box_zoom', 'wheel_zoom', 'reset']
    )

    # add limits to plot
    if x_range is not None:
        p.x_range = Range1d(start=x_range[0], end=x_range[1])

    legend_it = defaultdict(list)
    hist_renderers = []
    pdf_renderers = []

    grouped = df.groupby(group_by_values)
    num_groups = grouped.ngroups
    for i, (label, group) in enumerate(grouped):
        # Histogram
        values = group[x_label]
        density, edges = np.histogram(values, bins=bins, density=True)
        count, _ = np.histogram(values, bins=bins)
        pdf = norm.pdf(edges, np.mean(values), np.std(values))

        # Calculate edges
        edge_distance = edges[1] - edges[0]
        left_edges = edges[:-1] + i / num_groups * edge_distance
        right_edges = edges[1:] - (num_groups - i - 1) / num_groups * edge_distance

        hist_df = pd.DataFrame({
            'Density': density, 
            'Left_edge': edges[:-1], 
            'Right_edge': edges[1:], 
            'Left_plot_edge': left_edges, 
            'Right_plot_edge': right_edges, 
            'Count': count,
            'Total_count': np.full_like(density, len(group))
        })
        hist_source = ColumnDataSource(hist_df)

        pdf_df = pd.DataFrame({
            'Edge': edges,
            'PDF': pdf,
        })
        pdf_source = ColumnDataSource(pdf_df)

        # Plot histogram and PDF
        r_hist = p.quad(top='Density', bottom=0, left='Left_plot_edge', right='Right_plot_edge', source=hist_source, alpha=0.6, color=color_palette[i], line_color='white')
        r_pdf = p.line('Edge', 'PDF', source=pdf_source, line_width=5, color=color_palette[i])
        hist_renderers.append(r_hist)
        pdf_renderers.append(r_pdf)

        # Legend entries
        legend_label = legend_label_callable(label)
        legend_it[legend_label].extend([r_hist, r_pdf])

    # All PDFs infront of histograms
    p.renderers = hist_renderers + pdf_renderers

    # Legend
    legend_it = list(legend_it.items())
    legend = Legend(items=legend_it, orientation='horizontal', click_policy='hide')
    legend.ncols = calculate_ncols(figure_size[0], legend) if legend_cols is None else legend_cols
    p.add_layout(legend, 'above')
    p.height += legend.glyph_height + legend.label_height + 2*legend.spacing + 2*legend.margin

    # Hover tool
    hover_hist = HoverTool(
        tooltips = [
            (f'{x_label} bin edges', '@Left_edge, @Right_edge'),
            ('Count', '@Count / @Total_count'),
        ],
        renderers = hist_renderers,
    )
    p.add_tools(hover_hist)

    hover_pdf = HoverTool(
        tooltips = [
            ('Probability', '@PDF')
        ],
        renderers = pdf_renderers,
    )
    p.add_tools(hover_pdf)

    if latex_style:
        set_figure_to_default_latex(p, output_backend='svg')

    return p