from typing import List


def is_notebook():
    """ Checks whether the python interpreter is running within a jupyter notebook kernel. """
    
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def tune_report(**metrics):
    """ Reports metrics to Ray Tune, supporting both the old and the new reporting API.

    Ray 1.x takes metrics as keyword arguments, while Ray 2.x takes a single dictionary.
    """

    import inspect
    from ray import tune

    if 'metrics' in inspect.signature(tune.report).parameters:
        tune.report(metrics)
    else:
        tune.report(**metrics)
