import unittest
class TDD_TEST_CALCULUS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.foo = 1
    def test_test_calculus(self):
        self.assertEqual(self.foo,1)
if __name__ == '__main__':
    unittest.main()
                