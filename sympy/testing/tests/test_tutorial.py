import unittest
from sympy import *
class TDD_TUTORIAL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_tutorial(self):
        x,y=symbols('x,y')
        expr = cos(x) + 1
        expr.subs(x, y)
        # Evaluating an expression at a point
        s=expr.subs(x, 0)
        self.assertEqual(s, 2)
        expr = x**y
        self.assertEqual(expr, x ** y)
        # Replacing a subexpression with another subexpression
        expr = expr.subs(y, x**y)
        self.assertEqual(expr,x**(x**y))
        expr = expr.subs(y, x**x)
        self.assertEqual(expr, x ** (x ** (x ** x)))
        expr = sin(2*x) + cos(2*x)
        self.assertEqual(expand_trig(expr),2*sin(x)*cos(x) + 2*cos(x)**2 - 1)
        self.assertEqual(expr.subs(sin(2 * x), 2 * sin(x) * cos(x)), 2 * sin(x) * cos(x) + cos(2 * x))
        # SymPy objects are immutable
        expr = cos(x)
        self.assertEqual(expr.subs(x, 0),1)
        self.assertEqual(expr,cos(x))
        self.assertEqual(x, x)
    def test_subs(self):
        x,y,z=symbols('x,y,z')
        expr = x**3 + 4*x*y - z
        self.assertEqual(expr.subs([(x, 2), (y, 4), (z, 0)]),40)
        expr = x**4 - 4*x**3 + 4*x**2 - 2*x + 3
        replacements = [(x**i, y**i) for i in range(5) if i % 2 == 0]
        self.assertEqual(expr.subs(replacements),y**4-4*x**3+4*y**2-2*x+3)
        self.assertEqual(expr.subs(replacements),-4*x**3 - 2*x + y**4 + 4*y**2 + 3)
if __name__ == '__main__':
    unittest.main()
                