import sys
sys.path.append('../src/')

import pytest

from autodiff_team44.variable import *
from autodiff_team44.ad import *
from autodiff_team44.functions import *
import numpy as np
import math

class TestVariable:

    def test_neg(self):
        x = Variable(1, 1)
        y = -x
        assert y.val == -1
        assert y.der == -1

    def test_add(self):
        x = Variable(1, 1)
        z = x+2
        assert z.val == 3
        assert z.der == 1

    def test_add_variable(self):
        x = Variable(1,1)
        y = Variable(2,1)
        z = x+y
        assert z.val == 3
        assert z.der == 2

    def test_radd(self):
        x = Variable(1, 1)
        z = 2+x
        assert z.val == 3
        assert z.der == 1

    def test_sub(self):
        x = Variable(1, 1)
        z = x-2
        assert z.val == -1
        assert z.der == 1

    def test_sub_variable(self):
        x = Variable(1,1)
        y = Variable(2,1)
        z = x-y
        z2 = x - x
        assert z.val == -1
        assert z.der == 0

    def test_rsub(self):
        x = Variable(1, 1)
        z = 2-x
        assert z.val == 1
        assert z.der == -1

    def test_rsub_variable(self):
        x = Variable(1,1)
        y = Variable(2,1)
        z = y-x
        assert z.val == 1
        assert z.der == 0

    def test_mul(self):
        x = Variable(1, 1)
        z = x*2
        assert z.val == 2
        assert z.der == 2

    def test_mul_variable(self):
        x = Variable(1, 1)
        y = Variable(2, 1)
        z = x*y
        assert z.val == 2
        assert z.der == 3

    def test_rmul(self):
        x = Variable(1, 1)
        z = 2*x
        assert z.val == 2
        assert z.der == 2

    def test_rmul_variable(self):
        x = Variable(1, 1)
        y = Variable(2, 1)
        z = x*y
        assert z.val == 2
        assert z.der == 3

    def test_div(self):
        x = Variable(2, 1)
        z = x/2
        assert z.val == 1
        assert z.der == 1/2

    def test_div_variable(self):
        x = Variable(1, 1)
        y = Variable(2, 1)
        z = x/y
        assert z.val == 1/2
        assert z.der == 1/4

    def test_rdiv(self):
        x = Variable(2, 1)
        z = 2/x
        assert z.val == 1
        assert z.der == -1/2

    def test_rdiv_variable(self):
        x = Variable(1, 1)
        y = Variable(2, 1)
        z = x/y
        assert z.val == 1/2
        assert z.der == 1/4

    def test_pow(self):
        x = Variable(2,1)
        z = x**2
        assert z.val == 4
        assert z.der == 4

    def test_rpow(self):
        x = Variable(2,1)
        z = 2**x
        assert z.val == 4
        assert z.der == np.log(2) * 2 ** 2

    def test_pow_variable(self):
        x = Variable(2, 1)
        y = Variable(3, 1)
        z = x**y
        assert z.val == 2**3
        assert z.der == 2 ** 3 * 1 * np.log(2) + 1 * 3 * 2 ** (3 - 1)

    def test_str(self):
        x = Variable(1,2)
        y = repr(x)
        assert y == "f = 1\nf' = 2"

    def test_variable_number_invalid_val_type(self):
        # Try to create a Variable object with an invalid type for the val part
        with pytest.raises(TypeError):
            Variable("hello", 1)

    def test_variable_number_invalid_der_type(self):
        # Try to create a Variable object with an invalid type for the der part
        with pytest.raises(TypeError):
            Variable(2, "hello")

    def test_unequal_numpy_array_sizes(self):
        with pytest.raises(Exception):
            Variable(np.array([1, 2]), np.array([1, 2, 3]))

    # Test that an Exception is raised when the val part of the base is less than or equal to 0 and the exponent is a Variable
    def test_invalid_base_and_variable_exponent(self):
        with pytest.raises(Exception):
            Variable(-1, 2) ** Variable(3, 4)
        with pytest.raises(Exception):
            Variable(-1, 2) ** Variable(-3, -4)
        with pytest.raises(Exception):
            Variable(np.array([-1, -2]), np.array([2, 3])) ** Variable(np.array([3, 4]), np.array([4, 5]))

    # Test that the gradient method returns the correct gradient for a scalar Variable
    def test_gradient_scalar(self):
        assert Variable(1, 2).gradient() == 2
        assert Variable(1, -2).gradient() == -2

    # Test that the gradient method returns the correct gradient for a Variable with a numpy array as the val part
    def test_gradient_array(self):
        assert np.array_equal(Variable(np.array([1, 2]), np.array([2, 3])).gradient(), np.array([2, 3]))
        assert np.array_equal(Variable(np.array([1, 2]), np.array([-2, -3])).gradient(), np.array([-2, -3]))

    # Test that the derivative method returns the correct derivative for a scalar Variable
    def test_derivative_scalar(self):
        assert Variable(1, 2).derivative() == 2
        assert Variable(1, -2).derivative() == -2

    # Test that the derivative method returns the correct derivative for a Variable with a numpy array as the val part
    def test_derivative_array(self):
        assert np.array_equal(Variable(np.array([1, 2]), np.array([2, 3])).derivative(), np.array([2, 3]))
        assert np.array_equal(Variable(np.array([1, 2]), np.array([-2, -3])).derivative(), np.array([-2, -3]))

    def test_evaluate(self):
        #test for int
        x = Variable(3)
        assert x.evaluate() == 3

        #test for float
        y = Variable(3.5)
        assert y.evaluate() == 3.5

        #test for numpy array
        z = Variable(np.array([3, 4]))
        assert np.array_equal(z.evaluate(), np.array([3, 4]))

    def test_higher_order(self):
        #test for order = 1
        x = Variable(3)
        assert x.higher_order(1).der == 1

        #test for order = 2
        y = Variable(3)
        assert y.higher_order(2).der == 1

        #test for order = 3
        z = Variable(3)
        assert z.higher_order(3).der == 1

    def test_higher_order_typeError(self):
        #test for order is not an integer
        x = Variable(3)
        with pytest.raises(TypeError):
            x.higher_order(1.5)

    def test_higher_order_valueError(self):
        #test for order is less than 1
        y = Variable(3)
        with pytest.raises(TypeError):
            y.higher_order(0)

    def test_gradient(self):
        #test for int
        x = Variable(3)
        assert x.gradient() == 1

        #test for float
        y = Variable(3.5)
        assert y.gradient() == 1

        #test for numpy array
        z = Variable(np.array([3, 4]))
        assert np.array_equal(z.gradient(), np.array([1, 1]))

    def test_gradient_valueError(self):
        #test for exponentiation base is less than or equal to 0
        x = Variable(0)
        with pytest.raises(Exception):
            x.__pow__(Variable(2))
