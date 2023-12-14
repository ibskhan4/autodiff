import numpy as np
from .variable import Variable

#defining the trigonometric functions
def sin(var):
    """Adapts the sin function to work with Variable"""
    if isinstance(var, Variable):
        return Variable(sin(var.val), cos(var.val) * var.der)
    else:
        return np.sin(var)

def cos(var):
    """Adapts the cos function to work with Variable"""
    if isinstance(var, Variable):
        return Variable(cos(var.val), -sin(var.val) * var.der)
    else:
        return np.cos(var)

def tan(var):
    """Adapts the tan function to work with Variable"""
    if isinstance(var, Variable):
        return Variable(tan(var.val), 1 / (cos(var.val) ** 2) * var.der)
    else:
        return np.tan(var)

#defining exponential function
def exp(var):
    """Adapts the exp function to work with Variable"""
    if isinstance(var, Variable):
        return Variable(exp(var.val), exp(var.val) * var.der)
    else:
        return np.exp(var)

#defining inverse trigonometric functions
def arcsin(var):
    """Adapts the arcsin function to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(arcsin(var.val), (1/np.sqrt(1 - (var.val)**2)) * var.der)
    else:
        return np.arcsin(var)

def arccos(var):
    """Adapts the arccos function to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(arccos(var.val), (-1/np.sqrt(1 - (var.val)**2)) * var.der)
    else:
        return np.arccos(var)

def arctan(var):
    """Adapts the arctan function to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(arctan(var.val), (1/(1 + (var.val)**2)) * var.der)
    else:
        return np.arctan(var)

#defining hyperbolic functions
def sinh(var):
    """Adapts the sinh function to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(sinh(var.val), cosh(var.val) * var.der)
    else:
        return np.sinh(var)

def cosh(var):
    """Adapts the cosh functions to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(cosh(var.val), sinh(var.val) * var.der)
    else:
        return np.cosh(var)

def tanh(var):
    """Adapts the tanh functions to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(tanh(var.val), (1 - tanh(var.val)**2) * var.der)
    else:
        return np.tanh(var)

#defining the logistic function
def logis(var):
    """Adapts for application of logistic functions with Dual numbers"""
    if isinstance(var, Variable):
        return Variable(logis(var.val), np.exp(-var.val)/(1 + np.exp(-var.val)**2) * var.der)
    else:
        return (1 / (1 + np.exp(-var)))

#defining the logarithmic function
def log(var):
    """Adapts logarithmic functions for use with Dual numbers"""
    if isinstance(var, Variable):
        return Variable(log(var.val), (1/(var.val*np.ln(10))) * var.der)
    else:
        return np.log(var)

#defining the square root function
def sqrt(var):
    """Adapts the sqrt function to work with dual numbers"""
    if isinstance(var, Variable):
        return Variable(sqrt(var.val), (1/(2*sqrt(var.val)) * var.der))
    else:
        return np.sqrt(var)