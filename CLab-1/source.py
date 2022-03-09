import numpy as np

########################################################
## Complete functions in skeleton codes below
## following instructions in each function.
## Do not modify existing function name or inputs.
## Do not test your codes here - use main.py instead.
## You may use any built-in functions from NumPy.
## You may define and call new functions as you see fit.
########################################################


def low_rank_approx(A, k):
    '''
    inputs: 
      - A: m-by-n matrix
      - k: positive integer, k<=m, k<=n
    returns:
      X: m-by-n matrix that is an as-close-as-possible approximation of A
         up to rank k
    '''
    u, s, v = np.linalg.svd(A)
    s_X = np.diag(s[:k])
    u_X = u[:,:k]
    v_X = v[:k,:]
    X = u_X @ s_X @ v_X
    return X
    #TODO: Fill your work here
    

def constrained_LLS(A, B):
    '''
    inputs:
      - A: n-by-n full rank matrix
      - B: n-by-n matrix
    returns:
      x: n-diemsional vector that minimises ||Ax||2 subject to ||Bx||2=1
    '''
    n = min(B.shape[0], B.shape[1])
    rankB = np.linalg.matrix_rank(B)
    if rankB != n:
        small_constant = np.diag([0.0000001] * n)
        B = B + small_constant
    u, s, v = np.linalg.svd(A.T@A)
    w, q = np.linalg.eig(A)
    V = v[:, n-1]
    x = np.linalg.inv(B.T@B)@B.T@V
    return x
    #TODO: Fill your work here



### you can optionally write your own functions like below ###

# def my_func_name(input1, input2, ...):
#     do something
#     return ...