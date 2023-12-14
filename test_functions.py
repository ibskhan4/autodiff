import sys
sys.path.append('../src/')

import pytest

from autodiff_team44.variable import *
from autodiff_team44.ad import *
from autodiff_team44.functions import *
import numpy as np
import math

class TestFunctions:

    def test_exp(self):
        x = Variable(2, 1)
        z = exp(x)
        x_val = exp(5)
        assert z.val == np.exp(2)
        assert z.val == np.exp(2)
        assert x_val == np.exp(5)

    def test_sin(self):
        x = Variable(np.pi, 1)
        z = sin(x)
        x_val = sin(np.pi)
        assert z.val == np.sin(np.pi)
        assert z.der == np.cos(np.pi)
        assert x_val == np.sin(np.pi)

    def test_cos(self):
        x = Variable(np.pi, 1)
        z = cos(x)
        x_val = cos(4)
        assert z.val == np.cos(np.pi)
        assert z.der == -np.sin(np.pi)
        assert x_val == np.cos(4)

    def test_tan(self):
        x = Variable(np.pi/4, 1)
        z = tan(x)
        x_val = tan(5)
        assert z.val == np.tan(np.pi/4)
        assert z.der == 1/(np.cos(np.pi/4))**2
        assert x_val == np.tan(5)

    def test_logis(self):
        # Test logis with scalar input
        x = logis(Variable(0))
        assert x.val == 0.5
        assert x.der == 0.5

    def test_sqrt(self):
        # Test sqrt with scalar input
        x = sqrt(Variable(4))
        assert x.val == 2
        assert x.der == 0.25


    def test_sinh_scalar(self):
        #test with scalar input
        x = Variable(2, 3)
        assert sinh(x).val == np.sinh(2)
        assert sinh(x).der == np.cosh(2) * 3

    def test_tanh_scalar(self):
        #test with scalar input
        x = Variable(2, 3)
        assert tanh(x).val == np.tanh(2)
        assert tanh(x).der == (1 - np.tanh(2)**2) * 3

    def test_cosh_scalar(self):
        #test with scalar input
        x = Variable(2, 3)
        assert cosh(x).val == np.cosh(2)
        assert cosh(x).der == np.sinh(2) * 3

    def test_arctan_scalar(self):
        #test with scalar input
        x = Variable(2, 3)
        assert arctan(x).val == np.arctan(2)
        assert arctan(x).der == (1/(1 + (2)**2)) * 3

    #pytest tests for arcsin and arccos
    def test_arcsin(self):
        x = Variable(0.5, 1)
        assert arcsin(x).val == np.arcsin(0.5)
        assert arcsin(x).der == 1/(np.sqrt(1-0.5**2))

    def test_arccos(self):
        x = Variable(0.5, 1)
        assert arccos(x).val == np.arccos(0.5)
        assert arccos(x).der == -1/(np.sqrt(1-0.5**2))
