from __future__ import print_function   # makes this work for python2 and 3

import unittest
import dummy as du


class test_dummy(unittest.TestCase):

    def test_double(self):
        self.assertAlmostEqual(du.double(2), 4)

    def test_triple(self):
        self.assertAlmostEqual(du.triple(2), 6)

if __name__ == '__main__':
    unittest.main()