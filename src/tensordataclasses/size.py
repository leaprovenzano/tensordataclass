from typing import Type, NamedTuple, ClassVar, TypeVar, Union, Tuple

import torch

BaseT = TypeVar('BaseT', bound=NamedTuple)
SizeT = TypeVar('SizeT', bound='KeyedSize')


def as_size(x: Union[torch.Size, Tuple[int]]) -> torch.Size:
    if isinstance(x, torch.Size):
        return x
    return torch.Size(x)


class KeyedSize:

    _infotype: ClassVar[Type]

    @classmethod
    def from_keyedtensors(cls, *keyedtensors):
        return cls.__new__(cls, **{k: v.shape for k, v in keyedtensors})

    def __new__(cls, *args, **kwargs):
        return super().__new__(
            cls, *map(as_size, args), **{k: as_size(v) for k, v in kwargs.items()}
        )

    def numel(self):
        return self._infotype(*map(torch.Size.numel, self))


def keyedsize(name: str, basetype: Type[BaseT]) -> Type[SizeT]:
    """create a new KeyedSize type with name given basetype

    Args:
        name: the name for this new size type
        basetype: a named tuple type, will be used w/ KeyedSize mixin to create a new size type.

    Example:
        >>> import torch
        >>> from collections import namedtuple
        >>> from tensordataclasses.size import keyedsize
        >>>
        >>> MyInfo = namedtuple('MyInfo', ['a', 'b'])
        >>> MySize = keyedsize('MySize', MyInfo)
        >>>
        >>> size = MySize(a=(1, 4), b=(1, 5))
        >>> size
        MySize(a=torch.Size([1, 4]), b=torch.Size([1, 5]))

        >>> size.numel()
        MyInfo(a=4, b=5)
    """
    return type(name, (KeyedSize, basetype), {'_infotype': basetype})
