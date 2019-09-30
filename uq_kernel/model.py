"""
This module represents the standard interface that is set up for models in the
uncertainty quantification (UQ) framework.
"""
from abc import ABCMeta, abstractmethod
import numpy as np


class UQModel(metaclass=ABCMeta):
    """An abstract class representing features of a model in the UQ kernel"""

    @property
    @abstractmethod
    def cost(self):
        return NotImplementedError

    @abstractmethod
    def evaluate(self, inputs):
        return NotImplementedError
