import numpy as np
import numba
import pandas as pd


@numba.guvectorize('int32[:,:],int32[:],int32[:,:]', '(x,y),(m)->(x,m)')
def cal(x, per, out):
    for i in range(x.shape[0]):
        out[i] = np.percentile(x[i], per)


if __name__=='__main__':
    x = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], dtype=np.int32)
    per=np.array([25,50],dtype=np.int32)
    out = np.array([[0]*x.shape[0]]*per.shape[0])

    cal(x,per,out)
    print(pd.DataFrame(out))