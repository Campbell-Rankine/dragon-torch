from typing import Union, Literal
from abc import abstractmethod
from templates.base_classes import DragonBaseClass


class BaseMetric(DragonBaseClass):
    dtype: type

    @abstractmethod
    def get(self, X): ...
