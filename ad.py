import numpy as np
from .variable import Variable
from .functions import *

def jacobian(fns, x=None):
    """
    Returns the Jacobian given a function or a list of functions in the form of Variable objects

    Parameters
    ----------
    fns: NumPy array of Variable objects, list of Variable objects, a Variable object, or a callable function
    x: Variable object with value at which to evaluate the function(s) fns if fns is a callable function, optional
        required if fns is a callable function

    Returns
    -------
    jacobian: NumPy array representing the Jacobian
    """

    # Check what the input parameters' types are to return the appropriate result
    if callable(fns):
        if x is None:
            raise Exception("x cannot be none if fns is callable")
        if isinstance(x, np.ndarray):
            if all([isinstance(i, Variable) for i in x]):
                return fns(x).derivative()
            elif all([isinstance(i, (int, float, np.number)) for i in x]): # If the user did not create a Variable object x in advance, do it for them
                x = Variable(x)
                return fns(x).derivative()
            else:
                raise TypeError("x must be a Variable, int, or float, or a NumPy array of Variable, int, or float")
        elif isinstance(x, Variable):
            return fns(x).derivative()
        elif isinstance(x, (int, float, np.number)): # If the user did not create a Variable object x in advance, do it for them
            x = Variable(x)
            return fns(x).derivative()
        else:
            raise TypeError("x must be a Variable, int, or float, or a NumPy array of Variable, int, or float")
    # If the user input a single function fns or multiple
    elif isinstance(fns, np.ndarray) or isinstance(fns, list):
        if not all([isinstance(f, Variable) for f in fns]) and not all([callable(f) for f in fns]):
            raise TypeError("Parameter elements must all be Variable objects or all callable functions")
        if callable(fns[0]):
            if x is None:
                raise Exception("x cannot be none if fns is callable")
            if isinstance(x, np.ndarray):
                if all([isinstance(i, Variable) for i in x]):
                    return np.vstack([f(x).derivative() for f in fns])
                elif all([isinstance(i, (int, float, np.number)) for i in x]): # If the user did not create a Variable object x in advance, do it for them
                    x = Variable(x)
                    return np.vstack([f(x).derivative() for f in fns])
                else:
                    raise TypeError("x must be a Variable, int, or float, or a NumPy array of Variable, int, or float")
            elif isinstance(x, Variable):
                return np.vstack([f(x).derivative() for f in fns])
            elif isinstance(x, (int, float, np.number)): # If the user did not create a Variable object x in advance, do it for them
                x = Variable(x)
                return np.vstack([f(x).derivative() for f in fns])
            else:
                raise TypeError("x must be a Variable, int, or float, or a NumPy array of Variable, int, or float")
        else:
            return np.vstack([f.derivative() for f in fns])
    elif isinstance(fns, Variable):
        return fns.derivative()
    else:
        raise TypeError("Parameter must be a NumPy array of Variable objects, a list of Variable objects, or a Variable object")


def gd(fns, initial=0, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1):
    """
    Returns the local minimum given a callable function and optional parameters to tune the gradient descent

    Parameters
    ----------
    fns: callable function
    initial: int, float, or NumPy array of int or float, optional (default is 0)
    learn_rate: float, optional (default is 0.01)
    max_iters: int, optional (default is 10000)
    tol: float, optional (default is 0.000001)
    num_vars: int, optional (default is 1)

    Returns
    -------
    cur_x: int, float, or NumPy array of int or float representing the computed local minimum
    """

    if not callable(fns) and not all([callable(f) for f in fns]):
        raise TypeError("fns must be a callable function or a NumPy array of callable functions")
    if not isinstance(initial, (int, float, np.number, np.ndarray)):
        raise TypeError("initial must be a number or a NumPy array of numbers")
    if not isinstance(learn_rate, (float, np.floating)):
        raise TypeError("learn_rate must be a float")
    if not isinstance(max_iters, (int, np.integer)):
        raise TypeError("max_iters must be an int")
    if not isinstance(tol, (float, np.floating)):
        raise TypeError("tol must be a float")
    if not isinstance(num_vars, (int, np.integer)):
        raise TypeError("num_vars must be an int")

    # Check if only one function of multiple functions
    if callable(fns):
        iters = 0

        # Check if single or multiple variables
        if num_vars == 1:
            cur_x = initial
            previous_step_size = 1

            while previous_step_size > tol and iters < max_iters:
                prev_x = cur_x
                # Gradient Descent
                cur_x = cur_x - learn_rate * jacobian(fns, Variable(prev_x))
                previous_step_size = abs(cur_x - prev_x)
                iters += 1
        else:
            if isinstance(initial, np.ndarray):
                cur_x = initial
            else:
                cur_x = np.full(num_vars, initial)
            previous_step_size = np.ones(num_vars)

            while previous_step_size.any() > tol and iters < max_iters:
                prev_x = cur_x
                # Gradient Descent
                cur_x = cur_x - learn_rate * jacobian(fns, Variable(prev_x))
                previous_step_size = abs(cur_x - prev_x)
                iters += 1
        
        return cur_x
    else:
        xs = []
        for f in fns:
            iters = 0

            # Check if single or multiple variables
            if num_vars == 1:
                cur_x = initial
                previous_step_size = 1

                while previous_step_size > tol and iters < max_iters:
                    prev_x = cur_x
                    # Gradient Descent
                    cur_x = cur_x - learn_rate * jacobian(f, Variable(prev_x))
                    previous_step_size = abs(cur_x - prev_x)
                    iters += 1

                xs.append(cur_x)
            else:
                if isinstance(initial, np.ndarray):
                    cur_x = initial
                else:
                    cur_x = np.full(num_vars, initial)
                previous_step_size = np.ones(num_vars)

                while previous_step_size.any() > tol and iters < max_iters:
                    prev_x = cur_x
                    # Gradient Descent
                    cur_x = cur_x - learn_rate * jacobian(f, Variable(prev_x))
                    previous_step_size = abs(cur_x - prev_x)
                    iters += 1

                xs.append(cur_x)
        
        return np.array(xs)


def ga(fns, initial=0, learn_rate=0.01, max_iters=10000, tol=0.000001, num_vars=1):
    """
    Returns the local maximum given a callable function and optional parameters to tune the gradient ascent

    Parameters
    ----------
    fns: callable function
    initial: int, float, or NumPy array of int or float, optional (default is 0)
    learn_rate: float, optional (default is 0.01)
    max_iters: int, optional (default is 10000)
    tol: float, optional (default is 0.000001)
    num_vars: int, optional (default is 1)

    Returns
    -------
    cur_x: int, float, or NumPy array of int or float representing the computed local maximum
    """

    if not callable(fns) and not all([callable(f) for f in fns]):
        raise TypeError("fns must be a callable function or a NumPy array of callable functions")
    if not isinstance(initial, (int, float, np.number, np.ndarray)):
        raise TypeError("initial must be a number or a NumPy array of numbers")
    if not isinstance(learn_rate, (float, np.floating)):
        raise TypeError("learn_rate must be a float")
    if not isinstance(max_iters, (int, np.integer)):
        raise TypeError("max_iters must be an int")
    if not isinstance(tol, (float, np.floating)):
        raise TypeError("tol must be a float")
    if not isinstance(num_vars, (int, np.integer)):
        raise TypeError("num_vars must be an int")

    # Check if only one function of multiple functions
    if callable(fns):
        iters = 0

        # Check if single or multiple variables
        if num_vars == 1:
            cur_x = initial
            previous_step_size = 1

            while previous_step_size > tol and iters < max_iters:
                prev_x = cur_x
                # Gradient Descent
                cur_x = cur_x + learn_rate * jacobian(fns, Variable(prev_x))
                previous_step_size = abs(cur_x - prev_x)
                iters += 1
        else:
            if isinstance(initial, np.ndarray):
                cur_x = initial
            else:
                cur_x = np.full(num_vars, initial)
            previous_step_size = np.ones(num_vars)

            while previous_step_size.any() > tol and iters < max_iters:
                prev_x = cur_x
                # Gradient Descent
                cur_x = cur_x + learn_rate * jacobian(fns, Variable(prev_x))
                previous_step_size = abs(cur_x - prev_x)
                iters += 1
        
        return cur_x
    else:
        xs = []
        for f in fns:
            iters = 0

            # Check if single or multiple variables
            if num_vars == 1:
                cur_x = initial
                previous_step_size = 1

                while previous_step_size > tol and iters < max_iters:
                    prev_x = cur_x
                    # Gradient Descent
                    cur_x = cur_x + learn_rate * jacobian(f, Variable(prev_x))
                    previous_step_size = abs(cur_x - prev_x)
                    iters += 1

                xs.append(cur_x)
            else:
                if isinstance(initial, np.ndarray):
                    cur_x = initial
                else:
                    cur_x = np.full(num_vars, initial)
                previous_step_size = np.ones(num_vars)

                while previous_step_size.any() > tol and iters < max_iters:
                    prev_x = cur_x
                    # Gradient Descent
                    cur_x = cur_x + learn_rate * jacobian(f, Variable(prev_x))
                    previous_step_size = abs(cur_x - prev_x)
                    iters += 1

                xs.append(cur_x)
        
        return np.array(xs)

# Test code

# x = Variable(np.array([-5, 3.5]))

# z1 = (x[0] ** 3 * x[1] ** 2 - 4 * x[0] ** 2 + 6 * x[0] - 24) / (-x[0] * x[1])
# z2 = x[0] ** 4 + 2.5 * x[1]
# z3 = x[1] ** x[0]

# def z1(x):
#     return (x[0] ** 3 * x[1] ** 2 - 4 * x[0] ** 2 + 6 * x[0] - 24) / (-x[0] * x[1])

# def z2(x):
#     return x[0] ** 4 + 2.5 * x[1]

# def z3(x):
#     return x[1] ** x[0]

# f = z1
# # Correct result: [35.8686, -22.4857]
# print(jacobian(f, x))

# fns = np.array([z1, z2, z3])
# # Correct result:
# # [35.8686,  -22.4857 ]
# # [-500,      2.5     ]
# # [0.002385, -0.002720]
# print(jacobian(fns, np.array([-5, 3.5])))
# print(z2(x).derivative())

# x = Variable(3)
# y = x ** 2 + 1
# print(jacobian(y))

# def z1(x):
#     return 2*x[0]**3 + 3*x[0]**2 - 12*x[0] + 3*x[1]**3 + 4*x[1]**2 - 6*x[1]

# def z2(x):
#     return 3*x[0]**3 + 4*x[0]**2 - 13*x[0] + 4*x[1]**3 + 5*x[1]**2 - 7*x[1]

# def z3(x):
#     return x[1] ** x[0]

# print(gd(np.array([z1, z2]), num_vars=2))
# print(ga(np.array([z1, z2]), num_vars=2))
