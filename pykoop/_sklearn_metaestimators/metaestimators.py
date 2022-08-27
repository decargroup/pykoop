"""Utilities for meta-estimators.

Based on code from the ``scikit-learn`` project. Original authors of the file
are Joel Nothman and Andreas Mueller. Specifically, the original file is
``scikit-learn/sklearn/utils/metaestimators.py`` from commit ``cc8d84a``.

I wanted my interface to be as close to the ``scikit-learn``
:class:`sklearn.pipeline.Pipeline` class as possible, but I did not want to
rely on the internal :class:`sklearn.utils.metaestimators._BaseComposition`
class to do so. My compromise was to copy and adjust that code for my own uses,
while making sure the original attribution and license were present.

Distributed under the BSD-3-Clause License. See ``LICENSE`` in this directory
for the full license.
"""

import abc
from typing import Any, Dict, List

import numpy as np
import sklearn.base


class _BaseComposition(sklearn.base.BaseEstimator, metaclass=abc.ABCMeta):
    """Parameter management for classifiers composed of named estimators."""

    def _get_params(self, attr: str, deep: bool = True) -> Dict[str, Any]:
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        if not hasattr(estimators, '__iter__'):
            return out
        out.update(estimators)
        for name, estimator in estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out["%s__%s" % (name, key)] = value
        return out

    def _set_params(self, attr: str, **params) -> '_BaseComposition':
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if attr in params:
            setattr(self, attr, params.pop(attr))
        # 2. Step replacement
        items = getattr(self, attr)
        names = []
        if hasattr(items, '__iter__'):
            names, _ = zip(*items)
        for name in list(params.keys()):
            if "__" not in name and name in names:
                self._replace_estimator(attr, name, params.pop(name))
        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr: str, name: str, new_val: Any) -> None:
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def _validate_names(self, names: List[str]) -> None:
        if len(set(names)) != len(names):
            raise ValueError("Names provided are not unique: {0!r}".format(
                list(names)))
        conflict_names = set(names).intersection(self.get_params(deep=False))
        if conflict_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: {0!r}".
                format(sorted(conflict_names)))
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                "Estimator names must not contain __: got {0!r}".format(
                    invalid_names))
