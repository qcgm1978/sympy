import unittest
from sympy import *

class TDD_TEST_CALCULUS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x,y,z=symbols('x y z')
        cls.x = x
        cls.y = y
        cls.z=z
        init_printing(use_unicode=True)
    def test_calculus(self):
        self.assertIsInstance(self.x, Symbol)
    def test_Derivatives(self):
        x=self.x
        self.assertEqual(diff(cos(x), x),-sin(x))
        self.assertEqual(diff(exp(x**2), x),2*x*E**(x**2))
        self.assertEqual(diff(x**4, x, x, x),24*x)
        self.assertEqual(diff(x**4, x,3),24*x)
    def test_Integrals(self):
        self.assertEqual(integrate(cos(self.x), self.x),sin(self.x))
        self.assertEqual(integrate(exp(-self.x), (self.x, 0, oo)),1)
    def test_Limits(self):
        x=symbols('x')
        self.assertEqual(limit(sin(x)/x, x, 0),1)
    def test_Series_Expansion(self):
        x=symbols('x')
        expr = exp(sin(x))
        self.assertEqual(expr.series(x, 0, 4),1+x+x**2/2+O(x**4))
    def test_Finite_differences(self):
        x=symbols('x')
        f = Function('f')
        dfdx = f(x).diff(x)
        self.assertEqual(dfdx.as_finite_difference(),simplify('-f(x - 1/2) + f(x + 1/2)'))

if __name__ == '__main__':
    unittest.main()
                