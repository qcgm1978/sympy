import unittest,math
from sympy import *
import sympy
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
        self.assertEqual(expr.subs(replacements), -4 * x ** 3 - 2 * x + y ** 4 + 4 * y ** 2 + 3)
    def test_convert_str(self):
        x,y,z=symbols('x,y,z')
        str_expr = "x**2 + 3*x - 1/2"
        expr = sympify(str_expr)
        self.assertEqual(expr.subs(x, 2),9.5)
    def test_evalf(self):
        x,y,z=symbols('x,y,z')
        expr = sqrt(8)
        f = expr.evalf()
        self.assertAlmostEqual(f,2.8,1)
        self.assertIsInstance(f,sympy.core.numbers.Float)
        self.assertEqual(len(str(pi.evalf(100)))-1,100)
        expr = cos(2*x)
        c=expr.evalf(subs={x: 2.4})
        self.assertEqual(c,cos(2*2.4))
        # Sometimes there are roundoff errors smaller than the desired precision that remain after an expression is evaluated. Such numbers can be removed at the user’s discretion by setting the chop flag to True.

        one = cos(1)** 2 + sin(1)** 2
        self.assertNotEqual((one - 1).evalf(),0)
        self.assertEqual((one - 1).evalf(chop=True),0)
    def test_lambdify(self):
        import numpy 
        x,y,z=symbols('x,y,z')
        a = numpy.arange(10)
        expr = sin(x)
        #  converts the SymPy names to the names of the given numerical library, usually NumPy
        f = lambdify(x, expr, "numpy") 
        self.assertIsInstance((f(a) ),numpy.ndarray)
        f = lambdify(x, expr, "math")
        self.assertIsInstance(f(0.1), float)
        def mysin(x):
            """
            My sine. Note that this is only accurate for small x.
            """
            return x
        f = lambdify(x, expr, {"sin":mysin})
        self.assertIsInstance(f(0.1),float)
    def test_printing(self):
        init_printing()
        # init_session() 
    def test_expand(self):
        x,y,z=symbols('x,y,z')
        self.assertEqual(expand((x + 1)**2),x**2+2*x+ 1)
        self.assertEqual(expand((x + 2)*(x - 3)),x**2-x-6)
        self.assertEqual(expand((x + 1)*(x - 2) - (x - 1)*x),-2)
    def test_factor(self):
        x,y,z=symbols('x,y,z')
        self.assertEqual(factor(x**3 - x**2 + x - 1),(x-1)*(x**2+1))
        p = x**2*z + 4*x*y*z + 4*y**2*z
        self.assertEqual(factor(p),(x+2*y)**2*z)
        self.assertEqual(factor_list(p),(1, [(z, 1), (x + 2*y, 2)]))
        self.assertEqual(expand((cos(x) + sin(x))** 2), sin(x)** 2 + cos(x)** 2 + 2 * cos(x) * sin(x))
    def test_collect(self):
        x,y,z=symbols('x,y,z')
        expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
        collected_expr = collect(expr,x)
        self.assertEqual(collected_expr,x**3 + x**2*(2 - z) + x*(y + 1) - 3)
        self.assertEqual(collected_expr.coeff(x, 2),2-z)
    def test_cancel(self):
        x,y,z=symbols('x,y,z')
        self.assertEqual(cancel((x**2 + 2*x + 1)/(x**2 + x)),(x+1)/x)
        expr = 1/x + (3*x/2 - 2)/(x - 4)
        self.assertEqual(cancel(expr),(3*x**2  - 2*x - 8)/  (2*x**2  - 8*x))
        expr = (x*y**2 - 2*x*y*z + x*z**2 + y**2 - 2*y*z + z**2)/(x**2 - 1)
        c=cancel(expr)
        self.assertEqual(c, (y**2  - 2*y*z + z**2)/     (x - 1))
        f=factor(expr)
        self.assertEqual(f,(y-z)**2/(x-1))
    def test_apart(self):
        x,y,z=symbols('x,y,z')
        expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
        self.assertEqual(apart(expr),(2*x - 1)/(x**2 + x + 1) - 1/(x + 4) + 3/x)
    def test_trigonometric(self):
        x,y,z=symbols('x,y,z')
        self.assertEqual(cos(acos(x)),x)
        self.assertEqual(asin(1),pi/2)
    def test_trigsimp(self):
        x,y,z,e=symbols('x,y,z,e')
        self.assertEqual(trigsimp(sin(x)**2 + cos(x)**2),1)
        # self.assertEqual(simplify(trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4)),simplify(cos(4*x)/2+1/2))
        self.assertEqual(trigsimp(cosh(x)**2 + sinh(x)**2),cosh(2*x))
        hyperbolic_cosine = 1 / 2 * (E ** z + E ** (-z))
        hc = trigsimp(hyperbolic_cosine)
        self.assertEqual(hc,0.5*E**z + 0.5*E**(-z))
    def test_expand_trig(self):
        x,y,z,e=symbols('x,y,z,e')
        self.assertEqual(expand_trig(sin(x + y)),sin(x)*cos(y) + sin(y)*cos(x))
        self.assertEqual(expand_trig(tan(2*x)),2*tan(x)/(1 - tan (x)**2))
        self.assertEqual(trigsimp(sin(x)*cos(y) + sin(y)*cos(x)),sin(x+y))
    def test_powers(self):
        x, y = symbols('x y', positive=True)
        a, b = symbols('a b', real=True)
        z, t, c = symbols('z t c')
        self.assertTrue(sqrt(x) == x ** Rational(1, 2))
        self.assertEqual(powsimp(x**a*x**b),x**(a+b))
        self.assertEqual(powsimp(x**a*y**a),(x*y)**a)
        self.assertEqual(powsimp(t**c*z**c),t**c*z**c)
        self.assertEqual(powsimp(t**c*z**c, force=True),(t*z)**c)
        self.assertEqual((z*t)**2,z**2*t**2)
        self.assertEqual(sqrt(x*y),sqrt(x)*sqrt(y))
        self.assertEqual(powsimp(z**2*t**2),z**2*t**2)
        self.assertEqual(powsimp(sqrt(x)*sqrt(y)),sqrt(x)*sqrt(y))
        self.assertEqual(expand_power_exp(x**(a + b)),x**a*x**b)
        self.assertEqual(expand_power_base((x*y)**a),x**a*y**a)
        self.assertEqual(expand_power_base((z*t)**c),(z*t)**c)
        self.assertEqual(expand_power_base((z*t)**c, force=True),z**c*t**c)
        self.assertEqual(x**2*x**3,x**5)
        self.assertEqual(expand_power_exp(x**5),x**5)
    def test_powdenest(self):
        x,y,e,a,b=symbols('x,y,e,a,b',  positive=True)
        self.assertEqual(powdenest((x**a)**b),x**(a*b))
        z=symbols('z')
        self.assertEqual(powdenest((z**a)**b),(z**a)**b)
        self.assertEqual(powdenest((z**a)**b,force=True),z**(a*b))
    def test_exponentials_logorithms(self):
        x,y,e,a,b=symbols('x,y,e,a,b')
        self.assertEqual(log(x), ln(x))
        self.assertEqual(log(E**(x+2*pi*I)),log(exp(x)))
        self.assertEqual(exp(1),E)
    def test_expand_log(self):
        x,y,e,a,b,n=symbols('x,y,e,a,b,n',  positive=True)
        t,z=symbols('t,z')
        self.assertEqual(expand_log(log(x*y)),log(x)+log(y))
        self.assertEqual(expand_log(log(x/y)),log(x)-log(y))
        self.assertEqual(expand_log(log(x**2)),2*log(x))
        self.assertEqual(expand_log(log(x**n)),n*log(x))
        self.assertEqual(expand_log(log(z*t)),log(t*z))
        self.assertEqual(expand_log(log(z**2)),log(z**2))
        self.assertEqual(expand_log(log(z**2),force=True),2*log(z))
    def test_logcombine(self):
        x,y,e,a,b,n,z=symbols('x,y,e,a,b,n,z',  positive=True)
        self.assertEqual(logcombine(log(x) + log(y)),log(x*y))
        self.assertEqual(logcombine(n*log(x)),log(x**n))
        self.assertEqual(logcombine(n*log(z)),log(z**n))
        self.assertEqual(logcombine(n*log(z), force=True),log(z**n))
    def test_special_functions(self):
        x, y, z = symbols('x y z')
        k, m, n = symbols('k m n')
        num = 3
        f = factorial(num)
        self.assertEqual(f,6)
        self.assertEqual(binomial(num, 2),f/2)
        self.assertEqual(gamma(num+1),f)
        self.assertEqual(hyper([1, 2], [3], num),hyper((1, 2), (3,), 3))
    def test_rewrite(self):
        x, y, z = symbols('x y z')
        self.assertEqual(tan(x).rewrite(sin),2*sin(x)**2/sin(2*x))
        self.assertEqual(factorial(x).rewrite(gamma),gamma(x+1))
    def test_expand_func(self):
        x, y, z = symbols('x y z')
        self.assertEqual(expand_func(gamma(x + 3)),x*(x+1)*(x+2)*gamma(x))
    def test_hyperexpand(self):
        x, y, z = symbols('x y z')
        self.assertEqual(hyperexpand(hyper([1, 1], [2], z)),-log(1-z)/z)
        expr = meijerg([[1],[1]], [[1],[]], -z)
        self.assertEqual(hyperexpand(expr),E**(1/z))
    def test_combsimp(self):
        n, k = symbols('n k', integer = True)
        self.assertEqual(combsimp(factorial(n) / factorial(n - 3)), n * (n - 1) * (n - 2))
        self.assertEqual(combsimp(binomial(n+1, k+1)/binomial(n, k)),(n+1)/(k+1))
    def test_gammasimp(self):
        x, y, z = symbols('x y z')
        self.assertEqual(gammasimp(gamma(x)*gamma(1 - x)),pi/sin(pi*x))
    def test_continued_fractions(self):
        x, y, z = symbols('x y z')
        def list_to_frac(l):
            expr = Integer(0)
            for i in reversed(l[1:]):
                expr += i
                expr = 1/expr
            return l[0] + expr
        f=list_to_frac([x, y, z])
        self.assertEqual(f, x + 1 / (1 / z + y))
        self.assertEqual(list_to_frac([1, 2, 3, 4]),simplify('43/30'))
        syms = symbols('a0:5')
        a0, a1, a2, a3, a4 = syms
        fract = list_to_frac(syms)
        self.assertEqual(fract,a0+1/(a1+1/(a2+(1/(a3+1/a4)))))
        fract = cancel(fract)
        self.assertEqual(fract,(a0*a1*a2*a3*a4 + a0*a1*a2 + a0*a1*a4 + a0*a3*a4 + a0 + a2*a3*a4 + a2 + a4)/(a1*a2*a3*a4 + a1*a2 + a1*a4 + a3*a4 + 1))

if __name__ == '__main__':
    unittest.main()
                
