from typing import Type, Tuple
import random
import pytest
import torch
from tensordataclasses import tensordataclass
from tests.utils import tensor_equals
from dataclasses import dataclass


@dataclass
class TensorDataClassCase:

    klass: Type
    fields: Tuple[str]

    def gen_input(self):
        return {f: torch.rand(random.randint(1, 10)) for f in self.fields}


@tensordataclass
class Example:

    x: torch.Tensor
    y: torch.Tensor


@tensordataclass
class ChildExample(Example):

    z: torch.Tensor


@pytest.fixture(params=['basic_example_cls', 'inhereted_with_added_attr'])
def case(request):
    cases = {
        'basic_example_cls': TensorDataClassCase(Example, ('x', 'y')),
        'inhereted_with_added_attr': TensorDataClassCase(ChildExample, ('x', 'y', 'z')),
    }

    return cases[request.param]


def test_fields(case):
    assert case.klass._fields == case.klass._tensorfields == case.fields


def test_shape_name(case):
    assert case.klass._Size.__name__ == f'{case.klass.__name__}Size'


def test_shape_fields(case):
    assert case.klass._Size._fields == case.klass._fields


def test_size_and_shape(case):
    inp = case.gen_input()
    expected_shape = case.klass._Size(**{k: v.shape for k, v in inp.items()})

    inst = case.klass(**inp)
    assert inst.shape == expected_shape
    assert inst.size() == expected_shape


def test_keyedtensors(case):
    inp = case.gen_input()

    inst = case.klass(**inp)
    recreated = case.klass.from_keyedtensors(inst.keyedtensors())

    for field in case.fields:
        assert tensor_equals(getattr(inst, field), getattr(recreated, field))
