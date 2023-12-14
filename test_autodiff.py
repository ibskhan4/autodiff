import sys
sys.path.append('../src/')

import pytest

from autodiff_team44.variable import *
from autodiff_team44.ad import *
from autodiff_team44.functions import *
import numpy as np
import math

class TestAutodiff:

    def test_jacobian(self):

        x = Variable(np.array([-5, 3.5]))

        z1 = (x[0] ** 3 * x[1] ** 2 - 4 * x[0] ** 2 + 6 * x[0] - 24) / (-x[0] * x[1])
        z2 = x[0] ** 4 + 2.5 * x[1]
        z3 = x[1] ** x[0]

        fns = np.array([z1, z2, z3])

        y = jacobian(fns)

        assert np.allclose(y, np.array([[ 3.58685714e+01, -2.24857143e+01], [-5.00000000e+02,  2.50000000e+00],[ 2.38522134e-03, -2.71995512e-03]]))

    def test_derivative(self):
        x = Variable(np.array([1, 2]))
        y = x[0] + x[1]
        assert y.val == 3
        assert np.allclose(y.der, np.array([1., 1.]))

    def test_jacobiannew(self):
        # Create some Variable objects
        x1 = Variable(2, 1)
        x2 = Variable(3, 0)
        x3 = Variable(4, 0)

        # Compute the Jacobian of the input variables
        jac = jacobian([x1, x2, x3])

        # Verify that the Jacobian is correct
        assert jac[0] == 1
        assert jac[1] == 0
        assert jac[2] == 0

    def test_jacobian_single_function(self):
        # Create a Variable object
        x = Variable(2, 1)

        # Compute the Jacobian of the input variable
        jac = jacobian(x)

        # Verify that the Jacobian is correct
        assert jac == 1


    def test_jacobian_invalid_input_type(self):
        # Create some Variable objects
        x1 = Variable(2, 1)
        x2 = Variable(3, 0)
        x3 = Variable(4, 0)

        # Try to compute the Jacobian with an invalid input type
        with pytest.raises(TypeError):
            jacobian([x1, x2, x3, "hello"])

    def test_jacobian_invalid_input_type_onlyonestring(self):
        # Try to compute the Jacobian with an invalid input type
        with pytest.raises(TypeError):
            jacobian("hello")

    def test_jacobian_empty_input(self):
        # Try to compute the Jacobian with no input
        with pytest.raises(TypeError):
            jacobian()

    # Unit Tests for gd()
    def test_gd_fn_type(self):
        with pytest.raises(TypeError):
            gd("not a function")

    def test_gd_initial_type(self):
        with pytest.raises(TypeError):
            gd(lambda x: x**2, initial="not a number")

    def test_gd_learn_rate_type(self):
        with pytest.raises(TypeError):
            gd(lambda x: x**2, learn_rate="not a float")

    def test_gd_single_variable(self):
        def f(x):
            return x**2
        val = gd(f, initial=2, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1)
        assert val == 4.852279509612153e-05

    def test_gd_returns_expected_result_for_simple_function(self):
        def simple_function(x):
            return x**2
        assert gd(simple_function, initial=2, learn_rate=0.1, max_iters=5, tol=0.001, num_vars=1) == 0.65536

    def test_ga_function_invalid_function_input(self):
        """Test if ga function raises an error with an invalid function input"""
        with pytest.raises(TypeError):
            ga(1, 4, 0.1, 10000, 0.000001, 1)

    def test_ga_exception1(self):
        """Test that ga() raises TypeError when fn is not callable"""
        with pytest.raises(TypeError):
            ga(1)

    def test_ga_exception2(self):
        """Test that ga() raises TypeError when initial is not int/float/np.number/np.ndarray"""
        with pytest.raises(TypeError):
            ga(lambda x: x**2, initial="hello")

    def test_ga_exception3(self):
        """Test that ga() raises TypeError when learn_rate is not float"""
        with pytest.raises(TypeError):
            ga(lambda x: x**2, learn_rate=1)

    def test_ga_exception4(self):
        """Test that ga() raises TypeError when max_iter is not int"""
        with pytest.raises(TypeError):
            ga(lambda x: x**2, max_iter=0.5)

    def test_ga_exception5(self):
        """Test that ga() raises TypeError when tol is not float"""
        with pytest.raises(TypeError):
            ga(lambda x: x**2, tol=2)

    def test_ga_exception6(self):
        """Test that ga() raises TypeError when num_vars is not int"""
        with pytest.raises(TypeError):
            ga(lambda x: x**2, num_vars=0.5)

    def test_ga_convergence1(self):
        """Test that ga() converges for a single variable for the function f(x) = x^2"""
        assert ga(lambda x: x**2, initial=2) == 2.0079255910023497e+86

    def test_ga_convergence2(self):
        """Test that ga() converges for multiple variables for the function f(x, y) = x^2 + y^2"""
        assert list(ga(lambda x: x[0]**2 + x[1]**2, initial=np.array([2, 2]), num_vars=2)) == [2.0079255910023497e+86, 2.0079255910023497e+86]

    def test_gd_function_valid_input(self):
        """
        Tests that the gd function runs correctly with valid input
        """
        def fn(x):
            return x**2
        initial = 3
        learn_rate = 0.01
        max_iter = 10000
        tol = 0.000001
        num_vars = 1
        expected = 4.859130722701463e-05
        actual = gd(fn, initial, learn_rate, max_iter, tol, num_vars)
        assert(expected == actual)

    def test_gd_function_invalid_input(self):
        """
        Tests that the gd function raises an exception with invalid input
        """
        def fn(x):
            return x**2
        initial = "string"
        learn_rate = 0.01
        max_iter = 10000
        tol = 0.000001
        num_vars = 1
        with pytest.raises(TypeError):
            gd(fn, initial, learn_rate, max_iter, tol, num_vars)

    def test_jacobian_with_invalid_input_type(self):
        with pytest.raises(TypeError):
            jacobian(1)

    def test_jacobian_with_valid_callable_and_invalid_x(self):
        with pytest.raises(Exception):
            jacobian(lambda x: x**2)

    def test_jacobian_with_invalid_numpy_array_elements_in_x(self):
        with pytest.raises(TypeError):
            jacobian(lambda x: x**2, x=np.array([1, 'a']))

    def test_jacobian_with_invalid_fn_element_type(self):
        with pytest.raises(TypeError):
            jacobian([lambda x: x**2, 1], x=1)

    def test_jacobian_with_invalid_parameter_type(self):
        with pytest.raises(TypeError):
            jacobian(1, x=1)

    def test_jacobian_with_invalid_parameter_element_type(self):
        with pytest.raises(TypeError):
            jacobian([1, 2], x=1)

    def test_jacobian_type_error_1(self):
        with pytest.raises(TypeError):
            jacobian(1)

    def test_jacobian_type_error_2(self):
        with pytest.raises(TypeError):
            jacobian(np.array([1,2,3]), 1)

    def test_jacobian_type_error_3(self):
        with pytest.raises(TypeError):
            jacobian(np.array([1,2,3]), np.array([1,2,3]))

    def test_jacobian_type_error_5(self):
        with pytest.raises(TypeError):
            jacobian(np.array([1,2,3,4]))

    def test_jacobian_type_error_6(self):
        with pytest.raises(TypeError):
            jacobian(np.array([1,2,3,4]), np.array([1,2,3]))

    def test_jacobian_type_error_8(self):
        with pytest.raises(TypeError):
            jacobian(1, np.array([1,2,3]))

    def test_jacobian_function_and_variable(self):
        # Test jacobian with a callable function and a variable as input
        def fn(x):
            return x**2 + 1
        x = Variable(5)
        assert jacobian(fn, x) == 10

    def test_jacobian_function_and_number(self):
        # Test jacobian with a callable function and a number as input
        def fn(x):
            return x**2 + 1
        assert jacobian(fn, 5) == 10

    def test_jacobian_function_and_array(self):
        # Test jacobian with a callable function and an array of numbers as input
        def fn(x):
            return x**2 + 1
        assert np.allclose(jacobian(fn, np.array([1, 2, 3])),np.array([2, 4, 6]))


    def test_jacobian_callable_with_variable(self):
        """
        Test the jacobian function with a callable function and Variable object.
        """
        x = Variable(2)
        fns = lambda x: 2*x
        assert np.allclose(jacobian(fns, x), 2.0)


    def test_jacobian_variable(self):
        """
        Test the jacobian function with a single Variable object.
        """
        x = Variable(2)
        assert np.allclose(jacobian(x), 1.0)

    def test_jacobian_callable_with_int(self):
        """
        Test the jacobian function with a callable function and int.
        """
        x = 2
        fns = lambda x: 2*x
        assert np.allclose(jacobian(fns, x), 2.0)

    def test_jacobian_callable_with_float(self):
        """
        Test the jacobian function with a callable function and float.
        """
        x = 2.0
        fns = lambda x: 2*x
        assert np.allclose(jacobian(fns, x), 2.0)

    def test_jacobian_callable_with_numpy_number(self):
        """
        Test the jacobian function with a callable function and NumPy number.
        """
        x = np.int32(2)
        fns = lambda x: 2*x
        assert np.allclose(jacobian(fns, x), 2.0)

    def test_jacobian_callable_with_invalid_x(self):
        """
        Test the jacobian function with a callable function and invalid x.
        """
        x = '2'
        fns = lambda x: 2*x
        with pytest.raises(TypeError):
            jacobian(fns, x)

    def test_jacobian_list_invalid_elements(self):
        """
        Test the jacobian function with a list with invalid elements.
        """
        x = np.array([Variable(2), Variable(3)])
        fns = ['2*x[0]', 3*x[1]]
        with pytest.raises(TypeError):
            jacobian(fns)

    def test_jacobian_invalid_input(self):
        """
        Test the jacobian function with an invalid input.
        """
        x = '2'
        with pytest.raises(TypeError):
            jacobian(x)

    def test_gd_passing_function_parameter(self):
        """Test passing a valid callable function to gd()"""
        def test_func(x):
            return x ** 2
        assert callable(test_func)
        result = gd(test_func)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_initial_parameter(self):
        """Test passing a valid initial parameter to gd()"""
        def test_func(x):
            return x ** 2
        initial = 2
        assert isinstance(initial, (int, float, np.number, np.ndarray))
        result = gd(test_func, initial=initial)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_learn_rate_parameter(self):
        """Test passing a valid learn_rate parameter to gd()"""
        def test_func(x):
            return x ** 2
        learn_rate = 0.1
        assert isinstance(learn_rate, (float, np.floating))
        result = gd(test_func, learn_rate=learn_rate)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_max_iter_parameter(self):
        """Test passing a valid max_iter parameter to gd()"""
        def test_func(x):
            return x ** 2
        max_iter = 500
        assert isinstance(max_iter, (int, np.integer))
        result = gd(test_func, max_iters=max_iter)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_tol_parameter(self):
        """Test passing a valid tol parameter to gd()"""
        def test_func(x):
            return x ** 2
        tol = 0.001
        assert isinstance(tol, (float, np.floating))
        result = gd(test_func, tol=tol)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_num_vars_parameter(self):
        """Test passing a valid num_vars parameter to gd()"""
        def test_func(x):
            return x ** 2
        num_vars = 3
        assert isinstance(num_vars, (int, np.integer))
        result = gd(test_func, num_vars=num_vars)
        assert isinstance(result, (int, float, np.number, np.ndarray))

    def test_gd_passing_invalid_fn(self):
        """Test passing an invalid callable function to gd()"""
        invalid_fn = 'test'
        with pytest.raises(TypeError):
            gd(invalid_fn)

    def test_gd_passing_invalid_initial(self):
        """Test passing an invalid initial parameter to gd()"""
        def test_func(x):
            return x ** 2
        invalid_initial = 'test'
        with pytest.raises(TypeError):
            gd(test_func, initial=invalid_initial)

    def test_gd_passing_invalid_learn_rate(self):
        """Test passing an invalid learn_rate parameter to gd()"""
        def test_func(x):
            return x ** 2
        invalid_learn_rate = 'test'
        with pytest.raises(TypeError):
            gd(test_func, learn_rate=invalid_learn_rate)

    def test_gd_passing_invalid_max_iter(self):
        """Test passing an invalid max_iter parameter to gd()"""
        def test_func(x):
            return x ** 2
        invalid_max_iter = 'test'
        with pytest.raises(TypeError):
            gd(test_func, max_iter=invalid_max_iter)

    def test_gd_passing_invalid_tol(self):
        """Test passing an invalid tol parameter to gd()"""
        def test_func(x):
            return x ** 2
        invalid_tol = 'test'
        with pytest.raises(TypeError):
            gd(test_func, tol=invalid_tol)

    def test_gd_passing_invalid_num_vars(self):
        """Test passing an invalid num_vars parameter to gd()"""
        def test_func(x):
            return x ** 2
        invalid_num_vars = 'test'
        with pytest.raises(TypeError):
            gd(test_func, num_vars=invalid_num_vars)

    def test_jacobian_list_variable_objects(self):
        x = Variable(2)
        fns = [x**2, x+2]
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns), expected_jacobian)

    def test_jacobian_numpy_array_variable_objects(self):
        x = Variable(2)
        fns = np.array([x**2, x+2])
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns), expected_jacobian)

    def test_jacobian_list_callable_functions(self):
        x = Variable(2)
        fns = [lambda x: x**2, lambda x: x+2]
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns, x), expected_jacobian)

    def test_jacobian_numpy_array_callable_functions(self):
        x = Variable(2)
        fns = np.array([lambda x: x**2, lambda x: x+2])
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns, x), expected_jacobian)


    def test_jacobian_x_numpy_array_variable_objects(self):
        x = np.array([Variable(2), Variable(2)])
        fns = [lambda x: x[0]**2, lambda x: x[1] + 2]
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns, x), expected_jacobian)


    def test_jacobian_x_variable_object(self):
        x = Variable(2)
        fns = [lambda x: x**2, lambda x: x + 2]
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns, x), expected_jacobian)

    def test_jacobian_x_int_float_np_number(self):
        x = 2
        fns = [lambda x: x**2, lambda x: x + 2]
        expected_jacobian = np.array([[4], [1]])
        assert np.allclose(jacobian(fns, x), expected_jacobian)

    def test_jacobian_fns_not_variable_objects_or_callables(self):
        x = Variable(2)
        fns = [2, 3]
        with pytest.raises(TypeError):
            jacobian(fns, x)

    def test_jacobian_x_none(self):
        x = None
        fns = [lambda x: x**2, lambda x: x+2]
        with pytest.raises(Exception):
            jacobian(fns, x)

    def test_jacobian_x_not_variable_int_float_np_number(self):
        x = [1, 2]
        fns = [lambda x: x**2, lambda x: x+2]
        with pytest.raises(TypeError):
            jacobian(fns, x)

    def test_gd_single_variable(self):
        # Test a single variable function
        fns = [lambda x: x**2]
        initial = 0
        learn_rate = 0.01
        max_iters = 10000
        tol = 0.000001
        num_vars = 1
        expected_result = 0

        result = gd(fns, initial, learn_rate, max_iters, tol, num_vars)

        assert result == expected_result

    def test_gd_single_variable_non_default_params(self):
        # Test a single variable function with non-default params
        fns = [lambda x: x**2]
        initial = 0
        learn_rate = 0.1
        max_iters = 100
        tol = 0.001
        num_vars = 1
        expected_result = 0

        result = gd(fns, initial, learn_rate, max_iters, tol, num_vars)

        assert result == expected_result

def test_gd_single_variable():
    """
    Test the gd function with a single variable
    """
    f = lambda x: x**2
    x = gd(f, initial=2, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1)
    assert np.allclose(x, 0, atol=1e-4)

def test_gd_multiple_variables():
    """
    Test the gd function with multiple variables
    """
    f = lambda x: x[0]**2 + x[1]**2
    x = gd(f, initial=np.array([2,3]), learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=2)
    assert np.allclose(x, np.array([0, 0]), atol=1e-4)

def test_gd_multiple_functions():
    """
    Test the gd function with multiple functions
    """
    f1 = lambda x: x**2
    f2 = lambda x: x**3
    x = gd([f1, f2], initial=2, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1)
    assert np.allclose(x, np.array([4.85227951e-05, 5.77245990e-03]), atol=1e-4)

def test_ga_single_variable():
    """
    Test the ga function with a single variable
    """
    f = lambda x: x**2
    x = ga(f, initial=2, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1)
    assert np.allclose(x, 2.0079255910023497e+86, atol=1e-4)

def test_ga_multiple_variables():
    """
    Test the ga function with multiple variables
    """
    f = lambda x: x[0]**2 + x[1]**2
    x = ga(f, initial=np.array([2,3]), learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=2)
    assert np.allclose(x, np.array([2.00792559e+86, 3.01188839e+86]), atol=1e-4)

def test_ga_multiple_functions():
    """
    Test the ga function with multiple functions
    """
    f1 = lambda x: x**1
    f2 = lambda x: x**1
    x = ga([f1, f2], initial=2, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1)
    assert np.allclose(x, np.array([102., 102.]), atol=1e-4)
