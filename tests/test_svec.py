"""
test-svec.py

"""
# Copyright (c) 2012-24 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import os
import unittest
import collections
import warnings 
import copy
import math
import pickle
import numpy as np
# import gvar as gv
from dummy._svec_smat import *

FAST = False

if np.version.version >= '2.0':
    np.set_printoptions(legacy="1.25")

class ArrayTests(object):
    def __init__(self):
        pass

    def assert_gvclose(self,x,y,rtol=1e-5,atol=1e-8,prt=False):
        """ asserts that the means and sdevs of all x and y are close """
        if hasattr(x,'keys') and hasattr(y,'keys'):
            if sorted(x.keys())==sorted(y.keys()):
                for k in x:
                    self.assert_gvclose(x[k],y[k],rtol=rtol,atol=atol)
                return
            else:
                raise ValueError("x and y have mismatched keys")
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.asarray(x).flat
        y = np.asarray(y).flat
        if prt:
            print(np.array(x))
            print(np.array(y))
        for xi,yi in zip(x,y):
            self.assertGreater(atol+rtol*abs(yi.mean),abs(xi.mean-yi.mean))
            self.assertGreater(10*(atol+rtol*abs(yi.sdev)),abs(xi.sdev-yi.sdev))

    def assert_arraysclose(self,x,y,rtol=1e-5,prt=False):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        max_val = max(np.abs(list(x)+list(y)))
        max_rdiff = max(np.abs(x-y))/max_val
        if prt:
            print(x)
            print(y)
            print(max_val,max_rdiff,rtol)
        self.assertAlmostEqual(max_rdiff,0.0,delta=rtol)

    def assert_arraysequal(self,x,y):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = [float(xi) for xi in np.array(x).flatten()]
        y = [float(yi) for yi in np.array(y).flatten()]
        self.assertSequenceEqual(x,y)


class test_svec(unittest.TestCase,ArrayTests):
    def test_v(self):
        """ svec svec.assign svec.toarray """
        v = svec(3)   # [1,2,0,0,3]
        v.assign([1.,3.,2.],[0,4,1])
        self.assert_arraysequal(v.toarray(),[1.,2.,0.,0.,3.])

    def test_null_v(self):
        """ svec(0) """
        v = svec(0)
        self.assertEqual(len(v.toarray()),0)
        self.assertEqual(len(v.clone().toarray()),0)
        self.assertEqual(len(v.mul(10.).toarray()),0)
        u = svec(1)
        u.assign([1],[0])
        self.assertEqual(v.dot(u),0.0)
        self.assertEqual(u.dot(v),0.0)
        self.assert_arraysequal(u.add(v).toarray(),v.add(u).toarray())

    def test_v_clone(self):
        """ svec.clone """
        v1 = svec(3)   # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = v1.clone() # [0,10,0,0,20]
        self.assert_arraysequal(v1.toarray(),v2.toarray())
        v2.assign([10.,20.,30.],[0,1,2])
        self.assert_arraysequal(v2.toarray(),[10.,20.,30.])

    def test_v_dot(self):
        """ svec.dot """
        v1 = svec(3)   # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = svec(2)
        v2.assign([10.,20.],[1,4])
        self.assertEqual(v1.dot(v2),v2.dot(v1))
        self.assertEqual(v1.dot(v2),80.)
        v1 = svec(3)
        v1.assign([1,2,3],[0,1,2])
        v2 = svec(2)
        v2.assign([4,5],[3,4])
        self.assertEqual(v1.dot(v2),v2.dot(v1))
        self.assertEqual(v1.dot(v2),0.0)

    def test_v_add(self):
        """ svec.add """
        v1 = svec(3)    # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        v2 = svec(2)    # [0,10,0,0,20]
        v2.assign([10.,20.],[1,4])
        self.assert_arraysequal(v1.add(v2).toarray(),v2.add(v1).toarray())
        self.assert_arraysequal(v1.add(v2).toarray(),[1,12,0,0,23])
        self.assert_arraysequal(v1.add(v2,10,100).toarray(),[10.,1020.,0,0,2030.])
        self.assert_arraysequal(v2.add(v1,100,10).toarray(),[10.,1020.,0,0,2030.])
        v1 = svec(2)            # overlapping
        v1.assign([1,2],[0,1])
        v2.assign([3,4],[1,2])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,31.,28.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,31.,28.])
        v1 = svec(3)
        v2 = svec(3)
        v1.assign([1,2,3],[0,1,2])
        v2.assign([10,20,30],[1,2,3])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,80.,155.,210.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,80.,155.,210.])
        v1 = svec(2)
        v2 = svec(2)
        v1.assign([1,2],[0,1])  # non-overlapping
        v2.assign([3,4],[2,3])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,10.,21.,28.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,10.,21.,28.])
        v1 = svec(4)            # one encompasses the other
        v1.assign([1,2,3,4],[0,1,2,3])
        v2.assign([10,20],[1,2])
        self.assert_arraysequal(v1.add(v2,5,7).toarray(),[5.,80.,155.,20.])
        self.assert_arraysequal(v2.add(v1,7,5).toarray(),[5.,80.,155.,20.])

    def test_v_mul(self):
        """ svec.mul """
        v1 = svec(3)    # [1,2,0,0,3]
        v1.assign([1.,3.,2.],[0,4,1])
        self.assert_arraysequal(v1.mul(10).toarray(),[10,20,0,0,30])

    def test_pickle(self):
        v = svec(4)
        v.assign([1.,2.,5.,22], [3,5,1,0])
        with open('outputfile.p', 'wb') as ofile:
            pickle.dump(v, ofile)
        with open('outputfile.p', 'rb') as ifile:
            newv = pickle.load(ifile)
        self.assertEqual(type(v), type(newv))
        self.assertTrue(np.all(v.toarray() == newv.toarray()))
        os.remove('outputfile.p')

class test_smat(unittest.TestCase,ArrayTests):
    def setUp(self):
        """ make mats for tests """
        global smat_m,np_m
        smat_m = smat()
        smat_m.append_diag(np.array([0.,10.,200.]))
        smat_m.append_diag_m(np.array([[1.,2.],[2.,1.]]))
        smat_m.append_diag(np.array([4.,5.]))
        smat_m.append_diag_m(np.array([[3.]]))
        np_m = np.array([[ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,   10.,    0.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,    0.,  200.,    0.,    0.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    1.,    2.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    2.,    1.,    0.,    0.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    4.,    0.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    0.,    5.,    0.],
                       [   0.,    0.,    0.,    0.,    0.,    0.,    0.,    3.]])

    def tearDown(self):
        global smat_m,np_m
        smat_m = None
        np_m = None

    def test_m_append(self):
        """ smat.append_diag smat.append_diag_m smat.append_row smat.toarray"""
        self.assert_arraysequal(smat_m.toarray(),np_m)

    def test_m_dot(self):
        """ smat.dot """
        global smat_m,np_m
        v = svec(2)
        v.assign([10,100],[1,4])
        np_v = v.toarray()
        nv = len(np_v)
        self.assert_arraysequal(smat_m.dot(v).toarray(),np.dot(np_m[:nv,:nv],np_v))
        self.assert_arraysequal(smat_m.dot(v).toarray(),[0.,100.,0.,200.,100.])
        self.assertEqual(smat_m.dot(v).dot(v),np.dot(np.dot(np_m[:nv,:nv],np_v),np_v))
        self.assertEqual(smat_m.dot(v).size,3)

    def test_m_expval(self):
        """ smat.expval """
        global smat_m,np_m
        v = svec(2)
        v.assign([10.,100.],[1,4])
        np_v = v.toarray()
        nv = len(np_v)
        self.assertEqual(smat_m.expval(v),np.dot(np.dot(np_m[:nv,:nv],np_v),np_v))

    def test_pickle(self):
        """ pickle.dump(smat, outfile) """
        global smat_m
        with open('outputfile.p', 'wb') as ofile:
            pickle.dump(smat_m, ofile)
        with open('outputfile.p', 'rb') as ifile:
            m = pickle.load(ifile)
        self.assertEqual(type(smat_m), type(m))
        self.assertTrue(np.all(smat_m.toarray() == m.toarray()))
        os.remove('outputfile.p')

class test_smask(unittest.TestCase):
    def test_smask(self):
        def _test(imask):
            mask = smask(imask)
            np.testing.assert_array_equal(sum(imask[mask.starti:mask.stopi]), mask.len)
            np.testing.assert_array_equal(imask, np.asarray(mask.mask))
            np.testing.assert_array_equal(np.asarray(mask.map)[imask != 0], np.arange(mask.len))
            np.testing.assert_array_equal(np.cumsum(imask[imask != 0]) - 1, np.asarray(mask.map)[imask != 0])
        # g = gvar([1, 2, 3], [4, 5, 6])
        # gvar(1,0)
        # imask = np.array(g[0].der + g[2].der, dtype=np.int8)
        imask = np.array([1., 0., 1., 0.], dtype=np.int8)
        _test(imask)
    
    # def test_masked_ved(self):
    #     def _test(imask, g):
    #         mask = smask(imask)
    #         vec = g.internaldata[1].masked_vec(mask)
    #         np.testing.assert_array_equal(vec, g.der[imask!=0])
    #     g = gvar([1, 2, 3], [4, 5, 6])
    #     gvar(1,0)
    #     imask = np.array(g[0].der + g[1].der, dtype=np.int8)
    #     g[1:] += g[:-1]
    #     g2 = g**2
    #     _test(imask, g2[0])
    #     _test(imask, g2[1])
    #     _test(imask, g2[2])
    
    # def test_masked_mat(self):
    #     a = np.random.rand(3,3)
    #     g = gvar([1, 2, 3], a.dot(a.T))
    #     imask = np.array((g[0].der + g[2].der) != 0, dtype=np.int8)
    #     cov = evalcov([g[0], g[2]])
    #     gvar(1,0)
    #     mask = smask(imask)
    #     np.testing.assert_allclose(cov, g[1].cov.masked_mat(mask))

if __name__ == '__main__':
	unittest.main()

