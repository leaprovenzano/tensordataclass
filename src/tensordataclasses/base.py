from typing import Iterable, Tuple, ClassVar, Type
from collections import namedtuple
from dataclasses import dataclass

import torch

from tensordataclasses.size import keyedsize


class TensorDataClass:

    """Base class for all TensorDataClasses defining basic functionality

    Note:
        this class should not be used directly but will be added to bases when using the \
        tensordataclass decorator.
    """

    _Size: ClassVar[Type] = NotImplemented
    _Info: ClassVar[Type] = NotImplemented
    _fields: ClassVar[Tuple[str]] = NotImplemented

    @classmethod
    def from_keyedtensors(cls, *pairs):
        return cls(**dict(*pairs))

    def tensors(self) -> Iterable[torch.Tensor]:
        for field in self._fields:
            yield getattr(self, field)

    def keyedtensors(self) -> Iterable[Tuple[str, torch.Tensor]]:
        for field in self._fields:
            yield (field, getattr(self, field))

    def size(self):
        return self._Size.from_keyedtensors(*self.keyedtensors())

    @property
    def shape(self):
        return self.size()


def tensordataclass(cls=None, *args, **kwargs):
    """decorator for creating new tensordataclasses"""

    if cls is None:
        return lambda x: tensordataclass(cls=x, *args, **kwargs)
    else:
        if TensorDataClass not in cls.__mro__:
            cls = type(cls.__name__, (TensorDataClass,) + cls.__bases__, dict(cls.__dict__))

        tdc = dataclass(cls, *args, **kwargs)
        tdc._fields = tuple(f.name for f in tdc.__dataclass_fields__.values())

        def is_tensorfield(f: str) -> bool:
            return issubclass(tdc.__dataclass_fields__[f].type, torch.Tensor)

        tdc._tensorfields = tuple(filter(is_tensorfield, tdc._fields))

        tdc._Info = namedtuple(f'{tdc.__name__}Info', tdc._tensorfields)
        tdc._Size = keyedsize(f'{tdc.__name__}Size', tdc._Info)

        return tdc
