import numpy as np



def fr_fl(frac_str):
    """
    Converts a string input as fraction into floating point value
    """
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def r_ech(matrix):
    """
    converts input matrix into row echelon form and returns it.
    """
    mat = np.array(matrix)
    m,n = mat.shape
    for col in range(0, min(m,n)):
        if mat[col,col] == 0:
            for row in range(col, m):
                if mat[row, col] != 0:
                    mat[[col, row]] = mat[[row, col]]
                    break
        mat[col] = mat[col]/mat[col,col]
        for row in range(min(col+1, m), m):
            mat[row] = mat[row] - mat[row,col]*mat[col]
    return mat
            



def r_r_ech(matrix):
    """
    Converts input matrix into row reduced echelon form and returns it.
    """

def rank(self, matrix):
    """
    Returns rank of input matrix.
    """