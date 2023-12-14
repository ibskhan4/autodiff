import numpy as np

class ReverseVariable:
    def __init__(self, value, local_gradients = ()):
        """
        Initializes the ReverseVariable class with the variable's value (value) and children/their respective partial derivatives (local_gradients) components 
        stored as instance attributes

        value: int, float, or NumPy array
        local_gradients: tuple
        """
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        """Returns sum of ReverseVariables"""
        return add(self, other)

    def __mul__(self, other):
        """Returns product of ReverseVariables"""
        return mul(self, other)

    def __sub__(self, other):
        """Returns difference of ReverseVariables"""
        return add(self, negative(other))

    def __truediv__(self, other):
        """Returns quotient of ReverseVariables"""
        return mul(self, inv(other))

    def __rtruediv__(self, other):
        """Returns quotient of ReverseVariables with divisor and dividend reversed"""
        return mul(other, inv(self))

    def __neg__(self):
        """Returns negative of a ReverseVariables"""
        return negative(self)

    def __pow__(self, other):
        """Returns a ReverseVariable (self) taken to the power of another Reverse Variable (other) """
        return power(self, other)

    __radd__ = __add__

    __rmul__ = __mul__


def add(x, y):
    """Returns sum of two ReverseVariables"""
    value = x.value + y.value
    local_gradients = (
        (x, 1),
        (y, 1)
    )

    return ReverseVariable(value, local_gradients)

def mul(x, y):
    """Returns product of two ReverseVariables"""
    value = x.value * y.value
    local_gradients = (
        (x, y.value),
        (y, x.value)
    )

    return ReverseVariable(value, local_gradients)

def negative(x):
    """Returns negative of a ReverseVariable"""
    value = -1 * x.value
    local_gradients = (
        (x, -1),
    )

    return ReverseVariable(value, local_gradients)

def inv(x):
    """Returns inverse of a ReverseVariable"""
    value = 1. / x.value
    local_gradients = (
        (x, -1 / x.value**2),
    )

    return ReverseVariable(value, local_gradients)

def sin(x):
    """Returns sine of a ReverseVariable"""
    if isinstance(x, ReverseVariable):
        value = np.sin(x.value)
        local_gradients = (
            (x, np.cos(x.value)),
        )

        return ReverseVariable(value, local_gradients)
    else:
        return np.sin(x)
    

def cos(x):
    """Returns cosine of a ReverseVariable"""
    value = np.cos(x.value)
    local_gradients = (
        (x, -1*np.sin(x.value)),
    )

    return ReverseVariable(value, local_gradients)

def tan(x):
    """Returns tangent of a ReverseVariable"""
    value = np.tan(x.value)
    local_gradients = (
        (x, (1. / np.cos(x.value)) ** 2),
    )

    return ReverseVariable(value, local_gradients)

def exp(x):
    """Returns Euler's number to the power of a ReverseVariable"""
    value = np.exp(x.value)
    local_gradients = (
        (x, value),
    )

    return ReverseVariable(value, local_gradients)

def log(x):
    """Returns natural log of a ReverseVariable"""
    value = np.log(x.value)
    local_gradients = (
        (x, 1. / x.value),
    )

    return ReverseVariable(value, local_gradients)

def sqrt(x):
    """Returns square root of a ReverseVariable"""
    value = np.sqrt(x.value)
    local_gradients = (
        (x, 0.5 * (1. / x.value ** 2)),
    )

    return ReverseVariable(value, local_gradients)

def power(x, y):
    """Returns a Reverse Variable taken to the power of another ReverseVariable"""
    value = x.value ** y.value
    local_gradients = (
        (x, y.value * (x.value ** (y.value - 1))),
        (y, value * np.log(x.value)),
    )

    return ReverseVariable(value, local_gradients)





