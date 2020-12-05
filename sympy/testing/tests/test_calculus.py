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
    def test_test_calculus(self):
        self.assertIsInstance(self.x,Symbol)
if __name__ == '__main__':
    unittest.main()
                