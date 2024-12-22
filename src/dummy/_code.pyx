import numpy as np 
# cimport numpy as cnp 
# cnp.import_array()

def notdouble(double x):
    cdef Py_ssize_t i
    cdef Py_ssize_t[:] a = np.array([0,1,2], dtype=np.intp)
    for i in a:
        a[i] += 1
    return x + np.sum(a)
    
def triple(x):
    return 3*x
    