import os

from ipywidgets import widgets
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from bokeh.plotting import figure
from bokeh.io import export_svg, export_png
from typing import Literal


from .plotting import exclude_buttons_figure



def save_figures(all_plots: list[tuple[str, figure]], svg_results_dir: str, png_results_dir: str, executable_path='/snap/bin/firefox.geckodriver', save_format: Literal['svg', 'png', 'both'] = 'svg') -> None:
    """
    Saves all figures in a list with its corresponding name as png, svg or as both. Buttons and the bokeh logo will be removed.
     Only works with selenium geckodriver!

    Parameters
    ----------
    ``all_plots`` : list[tuple[str, figure]]
        A list of tuples where the first item is the filename and the second item is the figure to save.
    ``svg_results_dir`` : str
        The directory where the svg's should be stored.
    ``png_results_dir`` : str
        The directory where the png's should be stored.
    ``executable_path`` : str
        The path to the selenium geckodriver.
         Default = '/snap/bin/firefox.geckodriver'
    ``save_format`` : Literal['svg', 'png', 'both']
        Define in which formats the figures should be saved.
         Default = 'svg'
    """
    if not os.path.exists(svg_results_dir):
        os.makedirs(svg_results_dir)
    if not os.path.exists(png_results_dir):
        os.makedirs(png_results_dir)

    webdriver_service = Service(executable_path=executable_path)  # Path to geckodriver
    webdriver_firefox = webdriver.Firefox(service=webdriver_service)

    for name_and_plot in all_plots:
        if save_format in ['svg', 'both']:
            export_svg(exclude_buttons_figure(name_and_plot[1]), filename=os.path.join(svg_results_dir, f'{name_and_plot[0]}.svg'), webdriver=webdriver_firefox)
        if save_format in ['png', 'both']:
            export_png(exclude_buttons_figure(name_and_plot[1]), filename=os.path.join(png_results_dir, f'{name_and_plot[0]}.png'), webdriver=webdriver_firefox)

    webdriver_firefox.quit()



def save_figures_button(all_plots: list[tuple[str, figure]], svg_results_dir: str, png_results_dir: str, executable_path='/snap/bin/firefox.geckodriver'):
    """
    Creates a button, where one can choose the saving format. Needs to be pushed otherwise no action.
     Only works with selenium geckodriver!

    Parameters
    ----------
    ``all_plots`` : list[tuple[str, figure]]
        A list of tuples where the first item is the filename and the second item is the figure to save.
    ``svg_results_dir`` : str
        The directory where the svg's should be stored.
    ``png_results_dir`` : str
        The directory where the png's should be stored.
    ``executable_path`` : str
        The path to the selenium geckodriver.
         Default = '/snap/bin/firefox.geckodriver'

    Returns
    -------
    ``toggle_button`` : widgets.ToggleButtons
        ToggleButtons that show 'svg', 'png', 'both' as a saving format option.
    """
    toggle_button = widgets.ToggleButtons(
        value=None,
        description='Save Figures with format:',
        options=['svg', 'png', 'both'],
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Saves figures in svg format', 'Saves figures in png format', 'Saves figures in png and svg format'],
        # icon='check', # (FontAwesome names without the `fa-` prefix)
    )
    def on_change(toggle_value):
        value = toggle_value['new']
        if value is not None:
            save_figures(
                all_plots, 
                svg_results_dir,
                png_results_dir, 
                executable_path=executable_path, 
                save_format=value
            )
            toggle_button.value = None
    
    toggle_button.observe(on_change, names='value') 
    return toggle_button 
    