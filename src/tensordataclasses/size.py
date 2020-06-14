from typing import Type, NamedTuple, ClassVar, TypeVar

import torch

BaseT = TypeVar('BaseT', bound=NamedTuple)
SizeT = TypeVar('SizeT', bound='KeyedSize')


class KeyedSize:

    _infotype: ClassVar[Type]

    def __new__(cls, *args, **kwargs):
        return super().__new__(
            cls, *map(torch.Size, args), **{k: torch.Size(v) for k, v in kwargs.items()}
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
