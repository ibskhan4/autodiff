import numpy as np

class Variable:

    def __init__(self, val, der=None):
        """
        Initializes the Variable (Dual) class with evaluation (val) and derivative (der) components stored as instance attributes

        val: int, float, or NumPy array
        der: int, float, or NumPy array
        """
        # Make sure input parameters are valid
        if not isinstance(val, (int, float, np.number, np.ndarray, Variable)):
            raise TypeError("val part must be int, float, or NumPy array")
        elif not isinstance(der, (int, float, np.number, np.ndarray, Variable)) and der is not None:
            raise TypeError("der part must be int, float, or NumPy array")
        elif isinstance(val, np.ndarray):
            if der is not None:
                if not all([isinstance(r, (int, float, np.number, np.ndarray, Variable)) for r in val]) or not all([isinstance(d, (int, float, np.number, np.ndarray, Variable)) for d in der]):
                    raise TypeError("Elements of NumPy array must all be integers or floats")
                elif len(val) != len(der):
                    raise Exception("NumPy array sizes for val and der parts must be equivalent size")
            

        # If der part was empty, populate it accordingly based on what the val part was
        if der is None:
            self.val = val
            if isinstance(val, (int, float, np.number)):
                self.der = 1
            else:
                self.der = np.eye(len(val))
        else:
            self.val = val
            self.der = der

    def __getitem__(self, idx):
        """Returns the specified element to make the class subscriptable"""
        return Variable(self.val[idx], self.der[idx])

    def __len__(self):
        """Returns the length of the val part, useful for when it is a NumPy array"""
        return len(self.val)

    # Operator overloading with Variable object to return results where the val part is the evaluation and the der part is the derivative
    def __add__(self, other):
        """Returns sum of Variable and int/float or Variable and Variable"""
        if isinstance(other, Variable):
            val = self.val + other.val
            der = self.der + other.der
            return Variable(val, der)
        else:
            val = self.val + other
            der = self.der
            return Variable(val, der)

    def __radd__(self, other):
        """Returns sum of int/float and Variable"""
        return self + other

    def __mul__(self, other):
        """Returns product of Variable and int/float or Variable and Variable"""
        if isinstance(other, Variable):
            val = self.val * other.val
            der = self.val * other.der + self.der * other.val
            return Variable(val, der)
        else:
            val = self.val * other
            der = self.der * other
            return Variable(val, der)

    def __rmul__(self, other):
        """Returns product of int/float and Variable"""
        return self * other

    def __sub__(self, other):
        """Returns difference of Variable and int/float or Variable and Variable"""
        return self + (-1) * other

    def __rsub__(self, other):
        """Returns difference of int/float and Variable"""
        return other + (-1) * self

    def __truediv__(self, other):
        """Returns quotient of Variable and int/float or Variable and Variable"""
        return self * other ** float(-1)

    def __rtruediv__(self, other):
        """Returns difference of int/float and Variable"""
        return other * self ** float(-1)

    def __pow__(self, other):
        """Returns result of Variable taken to a power that is int/float or Variable"""
        if isinstance(other, Variable):
            if self.val <= 0:
                raise Exception("Exponentiation base cannot be less than or equal to 0 when calculating the derivative")

            val = self.val ** other.val
            der = self.val ** other.val * other.der * np.log(self.val) + self.der * other.val * self.val ** (other.val - 1)
            return Variable(val, der)
        else:
            val = self.val ** other
            der = other * self.der * (self.val ** (other - 1))
            return Variable(val, der)

    def __rpow__(self, other):
        """Returns result of int/float taken to a Variable"""
        # Prevent imaginary/complex results, we only want real numbers
        if other <= 0:
            raise Exception("Exponentiation base cannot be less than or equal to 0 when calculating the derivative")

        val = other ** self.val
        der = other ** self.val * self.der * np.log(other)
        return Variable(val, der)

    def __neg__(self):
        """Returns negation of Variable"""
        return -1 * self

    def __str__(self):
        """Returns val and der attributes in readable format displaying the function evaluation and derivative"""
        return f"f = {str(self.val)}\nf' = {str(self.der)}"

    def __repr__(self):
        """Returns val and der attributes in readable format displaying the function evaluation and derivative"""
        return f"f = {str(self.val)}\nf' = {str(self.der)}"

    def derivative(self):
        """Returns the derivative/the der instance attribute"""
        if isinstance(self.der, np.ndarray) and len(self.der) > 1:
            return self.gradient()

        der = self.der
        while isinstance(der, Variable):
            der = der.der

        return der

    def gradient(self):
        """Returns the gradient"""
        if not isinstance(self.der, np.ndarray) or len(self.der) < 2:
            return self.derivative()

        grad = []
        for i, der in enumerate(self.der):
            while isinstance(der, (Variable, np.ndarray)):
                try:
                    der = der[i]
                except:
                    try:
                        der = der.der
                    except:
                        raise Exception("Broken data structure")
            grad.append(der)

        return np.array(grad)

    def evaluate(self):
        """Returns the evaluation"""
        val = self.val
        while isinstance(val, Variable):
            val = val.val

        return val

    def higher_order(self, order):
        """Returns a new Variable object that can be used for higher order differentiation"""
        # Make sure the order parameter is an integer
        if not isinstance(order, int) or order < 1:
            raise TypeError("order parameter must be an integer greater than or equal to 1, or empty")

        new_var = Variable(self.val, self.der)

        for i in range(order - 1):
            new_var = Variable(new_var, new_var.der)

        return new_var


# Test code

# x = Variable(2.5, 1)
# y1 = x ** 3 - 4 * x ** 2 + 6 * x - 24
# # Correct result: f = -18.375, f' = 4.75
# print(y1)

# x = Variable(-5)
# y2 = (x ** 3 - 4 * x ** 2 + 6 * x - 24) / (-x)
# # Correct result: f = -55.8, f' = 13.04
# print(y2)

# y3 = 11 / -x
# # Correct result: f = 2.2, f' = 0.44
# print(y3)

# y4 = 11 - -x
# # Correct result: f = 6, f' = 1
# print(y4)

# x = Variable(-5, np.array([1, 0]))
# y = Variable(3.5, np.array([0, 1]))
# z1 = (x ** 3 * y ** 2 - 4 * x ** 2 + 6 * x - 24) / (-x * y)
# # Correct result: f = -96.3, fx = 35.8686, fy = -22.4857
# print(z1)

# x = Variable(np.array([-2, 5.5, 10]))
# y5 = (x[2] ** 2 * (x[0] * x[1] + 3) - 6 * (x[1] + 3)) / (x[0] * x[1] ** 2 + x[2] - 2)
# # Correct result: f = 16.2095, fx = -1.13642, fy = -2.86875, fz = 3.35637
# print(y5)

# x = Variable(np.array([1, 2]))
# y6 = 3 * x[0] ** 2 + x[1] ** 3
# print(y6.gradient())
# print(y6.evaluate())


# # Testing for higher order derivatives below
# x = Variable(2).higher_order(2)
# y = x
# print(y)

# x = Variable(np.array([1, 2, 3])).higher_order(2)
# y7 = (x[2] ** 3 - (x[1] + 3)) / (x[0] ** 2 + x[2] - 2)
# print(y7)
# # print("")
# # print(y7.evaluate())
