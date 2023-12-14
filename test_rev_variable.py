import sys
sys.path.append('../src/')

import pytest

from autodiff_team44.variable import *
from autodiff_team44.ad import *
from autodiff_team44.functions import *
from autodiff_team44.rev_variable import *
import numpy as np
import math

# Test __init__
def test_init_with_value():
    rev = ReverseVariable(2)
    assert rev.value == 2

def test_init_with_value_and_local_gradients():
    rev = ReverseVariable(2, (1, 2, 3))
    assert rev.value == 2
    assert rev.local_gradients == (1, 2, 3)

# Test __add__
def test_add_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = rev1 + rev2
    assert rev3.value == 5
    assert rev3.local_gradients == ((rev1, 1), (rev2, 1))

# Test __mul__
def test_mul_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = rev1 * rev2
    assert rev3.value == 6
    assert rev3.local_gradients == ((rev1, 3), (rev2, 2))


# Test __neg__
def test_neg_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = -rev1
    assert rev2.value == -2
    assert rev2.local_gradients == ((rev1, -1),)

# Test __pow__
def test_pow_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = rev1 ** rev2
    assert rev3.value == 8
    assert rev3.local_gradients == ((rev1, 3 * (2**(3 - 1))), (rev2, 8 * np.log(2)))


# Test add
def test_add_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = add(rev1, rev2)
    assert rev3.value == 5
    assert rev3.local_gradients == ((rev1, 1), (rev2, 1))

# Test mul
def test_mul_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = mul(rev1, rev2)
    assert rev3.value == 6
    assert rev3.local_gradients == ((rev1, 3), (rev2, 2))

# Test negative
def test_negative_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = negative(rev1)
    assert rev2.value == -2
    assert rev2.local_gradients == ((rev1, -1),)

# Test inv
def test_inv_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = inv(rev1)
    assert rev2.value == 1/2
    assert rev2.local_gradients == ((rev1, -1/(2**2)),)

# Test sin
def test_sin_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = sin(rev1)
    assert rev2.value == np.sin(2)
    assert rev2.local_gradients == ((rev1, np.cos(2)),)

def test_sin_with_int():
    result = sin(2)
    assert result == np.sin(2)

# Test cos
def test_cos_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = cos(rev1)
    assert rev2.value == np.cos(2)
    assert rev2.local_gradients == ((rev1, -1*np.sin(2)),)

# Test tan
def test_tan_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = tan(rev1)
    assert rev2.value == np.tan(2)
    assert rev2.local_gradients == ((rev1, (1. / np.cos(2)) ** 2),)

# Test exp
def test_exp_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = exp(rev1)
    assert rev2.value == np.exp(2)
    assert rev2.local_gradients == ((rev1, np.exp(2)),)

# Test log
def test_log_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = log(rev1)
    assert rev2.value == np.log(2)
    assert rev2.local_gradients == ((rev1, 1. / 2),)

# Test sqrt
def test_sqrt_with_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = sqrt(rev1)
    assert rev2.value == np.sqrt(2)
    assert rev2.local_gradients == ((rev1, 0.5 * (1. / 2 ** 2)),)

# Test power
def test_power_with_two_ReverseVariables():
    rev1 = ReverseVariable(2)
    rev2 = ReverseVariable(3)
    rev3 = power(rev1, rev2)
    assert rev3.value == 8
    assert rev3.local_gradients == ((rev1, 3 * (2**(3 - 1))), (rev2, 8 * np.log(2)))
