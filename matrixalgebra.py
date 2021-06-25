"""
Module to implement operations relating to systems of linear equations and matrices.

Classes:
-------
1. Elementary Operation calculator: Interactively performs elementary operations on matrix A of shape (m,n).
    call: ElementaryOps()
2. Transforming a matrix to row echelon form: Transforms matrix A of shape (m,n) to its row echelon or row reduced echelon form using ERO.
    call: RowEchelon()
3. Calculating invers: Calculates inverse of matrix A of shape (n,n) using ERO (if possible).
    call: Inverse()
4. Solve system of linear equations: Solves system of linear equations in m eqautions and n variables.
    call: SolveSystem()
"""

import numpy as np
import linalgtoolkit.__utils__ as util


class ElementaryOps(object):
    """
    Interactively performs elementary row operations on matrix A of shape (m,n).
    Supports two types of operations:
    1. Only row ops
    2. Congruence ops. (needs square matrix)

    Supports two modes of operations:
    1. Interactive, output directly printed.
    2. Independent, much more flexible. Returns matrix.

    Parameters:
    ----------
    matrix: array-like, list of lists of shape (m,n).
        can or cannot be numpy array.

    type: str. Possible values: {'row','congruence'}. Default='row'
        Determines which type of elementary operations to be performed. 

    verbose: bool. Determines the verbosity. Default False.
        At True, prints the matrix with additional info.
        At False, simply returns matrix.
    """

    def __init__(self, matrix, type: str = 'row', verbose: bool = False) -> None:
        self.matrix = np.array(matrix, dtype=float)
        self.m, self.n = self.matrix.shape
        if type not in ['row', 'congruence']:
            raise ValueError("type must be one of ['row', 'congurence']")
        if type == 'congruence' and (self.matrix!=self.matrix.transpose()).any():
            raise ValueError("Congruence operations can only be performed on symmetric matrices.")
        self.type = type
        self.verbose = verbose

    def operate(self, args: list=None):
        """
        args: (optional) list. Considered only when verbose='True'.
        Since the operations covered are elementary row and congurence, only row operations need to be taken as input.
        1.  add c times j-th row to i-th row; input: [1, c (float), j (int), i (int)]
        2.  multiply i-th row by c; input: [2, i (int), c (float)]
        3.  interchange i-th and j-th row; input: [3, i (int), j (int)]
        """
        if self.verbose == True:
            print("Matrix:")
            print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")
        if self.type == 'row':
            self.__row__(args)
        else:
            self.__cong__(args)

    def __row__(self, args):
        s = ("1. add c times j-th row to i-th row;\t input: 1 c j i\n"
            "2. multiply i-th row by c;\t input: 2 i c\n"
            "3. interchange i-th and j-th rows;\t input: 3 i j\n"
            "Input 'end' to stop.\n"
            "(note: replace letters with values.)\n")
        
        while True:
            if self.verbose == True:
                var = input(s).split()
            else:
                var = args
            if var[0] == 'end':
                break
            if str(var[0]) == '1':
                _,c,j,i= var
                c = util.fr_fl(c); j = int(j); i = int(i)
                self.matrix[i-1] = self.matrix[i-1] + c*self.matrix[j-1]
                if self.verbose == True:
                    print(f"R'{i} = R{i} + {c}*R{j}")
                    print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix
            elif str(var[0]) == '2':
                _,i,c = var
                c = util.fr_fl(c); i = int(i)
                self.matrix[i-1] = c*self.matrix[i-1]
                if self.verbose == True:
                    print(f"R'{i} = {c}*R{i}")
                    print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix
            elif str(var[0]) == '3':
                _,i,j = var
                i = int(i); j = int(j)
                self.matrix[[i-1, j-1]] = self.matrix[[j-1, i-1]]
                if self.verbose == True:
                    print(f"R{i} <-> R{j}")
                    print('\n'.join(['\t'.join([str(round(cell)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix

    def __cong__(self, args):
        s = ("1. add c times j-th row/column to i-th row/column;\t input: 1 c j i\n"
            "2. multiply i-th row/column by c;\t input: 2 i c\n"
            "3. interchange i-th and j-th rows/columns;\t input: 3 i j\n"
            "Input 'end' to stop.\n"
            "(note: replace letters with values.)\n")
        while True:
            if self.verbose == True:
                var = input(s).split()
            else:
                var = args
            if var[0] == 'end':
                break
            if str(var[0]) == '1':
                _,c,j,i= var
                c = util.fr_fl(c); j = int(j); i = int(i)
                self.matrix[i-1] = self.matrix[i-1] + c*self.matrix[j-1]
                self.matrix[:,[i-1]] = self.matrix[:,[i-1]] + c*self.matrix[:,[j-1]]
                if self.verbose == True:
                    print(f"R'{i} = R{i} + {c}*R{j}\nC'{i} = C{i} + {c}*C{j}")
                    print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix
            elif str(var[0]) == '2':
                _,i,c = var
                c = util.fr_fl(c); i = int(i)
                self.matrix[i-1] = c*self.matrix[i-1]
                self.matrix[:,[i-1]] = c*self.matrix[:,[i-1]]
                if self.verbose == True:
                    print(f"R'{i} = {c}*R{i}\nC'{i} = {c}*C{i}")
                    print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix
            elif str(var[0]) == '3':
                _,i,j = var
                i = int(i); j = int(j)
                self.matrix[[i-1, j-1]] = self.matrix[[j-1, i-1]]
                self.matrix[:, [i-1, j-1]] = self.matrix[:, [j-1, i-1]]
                if self.verbose == True:
                    print(f"R{i} <-> R{j}\nC{i} <-> C{j}")
                    print('\n'.join(['\t'.join([str(round(cell)) for cell in row]) for row in self.matrix]) + "\n")
                else:
                    return self.matrix

class RowEchelon(object):
    def __init__(self, matrix, verbose: bool = True) -> None:
        self.matrix = np.array(matrix, dtype=float)
        self.m, self.n = self.matrix.shape
        self.verbose = verbose

    def reduce(self):
        print("Matrix:")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")
        for col in range(0, min(self.m,self.n)):
            if self.matrix[col,col] == 0:
                for row in range(col, self.m):
                    if self.matrix[row, col] != 0:
                        self.matrix[[col, row]] = self.matrix[[row, col]]
                        print(f"R{col} <-> R{row}")
                        print('\n'.join(['\t'.join([str(round(cell)) for cell in row]) for row in self.matrix]) + "\n")
                        break
            
            print(f"R'{col} = R{col}/{self.matrix[col,col]}")
            self.matrix[col] = self.matrix[col]/self.matrix[col,col]
            print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
            for row in range(min(col+1, self.m), self.m):
                print(f"R'{row} = R{row} + {self.matrix[row,col]}*R{col}")
                self.matrix[row] = self.matrix[row] - self.matrix[row,col]*self.matrix[col]
                print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")

# class Inverse(object)

# class SolveSystem(object)
