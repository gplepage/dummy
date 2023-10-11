import numpy as np 
cimport numpy as cnp 
cnp.import_array()

def double(x):
    cdef cnp.npy_intp i
    a = np.array([0,1,2], dtype=np.intp)
    for i in a:
        a[i] += 1
    return 2*x
    
def triple(x):
    return 3*x
    