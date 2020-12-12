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
    def test_solvers(self):
        x, y, z = symbols('x y z')
        init_printing(use_unicode=True)
        eq_roots = roots
        eq_roots = {-1,1}
        self.assertEqual(solveset(Eq(x**2, 1), x),eq_roots)
        self.assertEqual(solveset(Eq(x**2 - 1, 0), x),eq_roots)
        self.assertEqual(solveset(x**2 - 1, x),eq_roots)
    def test_Solving_Equations_Algebraically(self):
        x, y, z = symbols('x y z')
        self.assertEqual(solveset(x**2 - x, x),{0,1})
        self.assertEqual(solveset(x - x, x, domain=S.Reals), S.Reals)
        a_root = root
        a_root = solveset(sin(x) - 1, x, domain=S.Reals)
        # self.assertEqual(root,2*Integers*pi+pi/2)
        self.assertEqual(solveset(exp(x), x) ,EmptySet)
        self.assertEqual(solveset(cos(x) - x, x),ConditionSet(x, Eq(-x + cos(x), 0), Complexes))  # Not able to find solution
        root_lin = {(-y - 1, y, 2)}
        self.assertEqual(linsolve([x + y + z - 1, x + y + 2*z - 3 ], (x, y, z)),root_lin)
        self.assertEqual(linsolve(Matrix(([1, 1, 1, 1], [1, 1, 2, 3])), (x, y, z)),root_lin)
        M = Matrix(((1, 1, 1, 1), (1, 1, 2, 3)))
        system = A, b = M[:, :-1], M[:, -1]
        self.assertEqual(linsolve(system, x, y, z),root_lin)
        a, b, c, d = symbols('a, b, c, d', real=True)
        self.assertEqual(nonlinsolve([a**2 + a, a - b], [a, b]),{(-1, -1), (0, 0)})
    def test_Matrics(self):
        M = Matrix([[1, 2, 3], [3, 2, 1]])
        N = Matrix([0, 1, 1])
        self.assertEqual(M*N,Matrix([[5],[3]]))
    def test_shape(self):
        M = Matrix([[1, 2, 3], [-2, 0, 4]])
        self.assertEqual(M.shape,(2,3))
        self.assertEqual(M.row(0),Matrix([[1,2,3]]))
        self.assertEqual(M.col(-1),Matrix([3,4]))
        M.col_del(0)
        self.assertEqual(M,Matrix([[2,3],[0,4]]))
        M.row_del(1)
        self.assertEqual(M,Matrix([[2,3]]))
        M = M.row_insert(1, Matrix([[0, 4]]))
        self.assertEqual(M,Matrix([[2,3],[0,4]]))
        M = M.col_insert(0, Matrix([1, -2]))
        self.assertEqual(M,Matrix([[1,2,3],[-2,0,4]]))
    def test_basic_methods(self):
        M = Matrix([[1, 3], [-2, 3]])
        N = Matrix([[0, 3], [0, 7]])
        self.assertEqual(M + N,Matrix([[1,6],[-2,10]]))
        self.assertEqual(M*N,Matrix([[0,24],[0,15]]))
        self.assertEqual(M**2,Matrix([[-5,12],[-8,3]]))
        self.assertEqual(M**-1,simplify('1/(1*3-(-2*3))* Matrix([[3,-3],[2,1]])'))


if __name__ == '__main__':
    unittest.main()
                
