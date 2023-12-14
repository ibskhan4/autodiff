import sys
sys.path.append('../src/')

import pytest

from autodiff_team44.variable import *
from autodiff_team44.ad import *
from autodiff_team44.functions import *
from autodiff_team44.rev_variable import *
from autodiff_team44.rev_ad import *
import numpy as np
import math


def test_find_gradients():
    x = ReverseVariable(3)  # x = 3
    y = ReverseVariable(4)  # y = 4
    f = (x + y) * x  # f = (3 + 4) * 3 = 21
    grads = find_gradients(f)
    assert grads == [10, 3] or [3, 10]

def test_rev_jacobian():
    x = ReverseVariable(3)  # x = 3
    y = ReverseVariable(4)  # y = 4
    f1 = (x + y) * x  # f1 = (3 + 4) * 3 = 21
    f2 = x * y  # f2 = 3 * 4 = 12
    fn_list = [f1, f2]
    jacobian = rev_jacobian(fn_list)
    assert np.array_equal(jacobian, [[3, 10], [3, 4]]) or np.array_equal(jacobian, [[10, 3], [4, 3]])

def test_rev_jacobian_exception():
    x = ReverseVariable(3)  # x = 3
    y = 4  # y = 4
    fn_list = [x, y]
    with pytest.raises(TypeError):
        rev_jacobian(fn_list)
