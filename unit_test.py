import unittest

def sum(a, b):
    return a + b

class SumTest(unittest.TestCase):

    def setUp(self):
        print("SETUP Called...")
        self.a = 10
        self.b = 20

    def test_sumfunc_1(self):
        print("TEST - 1 Called...")

        result = sum(self.a, self.b) + 1
        try:
            self.assertEqual(result, self.a + self.b, f"Excpected: {self.a + self.b}, got {result}")
        except AssertionError as e:
            print(e)

if __name__ == "__main__":
    unittest.main()
