"""Global configuration for ``pykoop``.

Based on code from the ``scikit-learn`` project. Original author of the file
is Joel Nothman. Specifically, the original file is
``scikit-learn/sklearn/_config.py`` from commit ``894b335``.

Distributed under the BSD-3-Clause License. See ``LICENSE`` in this directory
for the full license.
"""

import contextlib
import os
import threading
from typing import Any, Dict, Optional

_global_config = {
    'skip_validation': False,
}
_threadlocal = threading.local()


def _get_threadlocal_config() -> Dict[str, Any]:
    """Get a threadlocal mutable configuration.

    If the configuration does not exist, copy the default global configuration.
    """
    if not hasattr(_threadlocal, 'global_config'):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config() -> Dict[str, Any]:
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    Examples
    --------
    Get configuation

    >>> pykoop.get_config()
    {'skip_validation': False}
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(skip_validation: Optional[bool] = None) -> None:
    """Set global configuration.

    Parameters
    ----------
    skip_validation : Optional[bool]
        Set to ``True`` to skip all parameter validation. Can save significant
        time, especially in func:`pykoop.predict_trajectory()` but risks
        crashes.

    Examples
    --------
    Set configuation

    >>> pykoop.set_config(skip_validation=False)
    """
    local_config = _get_threadlocal_config()
    # Set parameters
    if skip_validation is not None:
        local_config['skip_validation'] = skip_validation


@contextlib.contextmanager
def config_context(*, skip_validation=None):
    """Context manager for global configuration.

    Parameters
    ----------
    skip_validation : Optional[bool]
        Set to ``True`` to skip all parameter validation. Can save significant
        time, especially in func:`pykoop.predict_trajectory()` but risks
        crashes.

    Examples
    --------
    Use config context manager

    >>> with pykoop.config_context(skip_validation=False):
    ...     pykoop.KoopmanPipeline()
    KoopmanPipeline()
    """
    old_config = get_config()
    set_config(skip_validation=skip_validation)

    try:
        yield
    finally:
        set_config(**old_config)
