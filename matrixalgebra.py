"""
Module to implement operations relating to systems of linear equations and matrices.

Classes:
-------
1. Elementary Operation calculator: Interactively performs elementary operations on matrix A of shape (m,n).
    call: ElementaryOps()
2. Transforming a matrix to row echelon form: Transforms matrix A of shape (m,n) to its row echelon form using ERO.
    call: RowEchelon()
3. Transforming a matrix to row reduced echelon form: Transforms matrix A of shape (m,n) to its row reduced echelon form using ERO.
    call: RowReducedEchelon()
4. Calculating invers: Calculates inverse of matrix A of shape (n,n) using ERO (if possible).
    call: Inverse()
5. Solve system of linear equations: Solves system of linear equations in m eqautions and n variables.
    call: SolveSystem()
"""

import numpy as np

class _utilities(object):
    def __init__(self, matrix) -> None:
        self.matrix = np.array(matrix)

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

    mode: str. Possible values: {'interactive','independent'}. Default='interactive'
        Determines the mode of use.

    args: (optional) list.
        Since the operations covered are elementary row and congurence, only row operations need to be taken as input.
    1.  add c times j-th row to i-th row; input: [1, c (float), j (int), i (int)]
    2.  multiply i-th row by c; input: [2, i (int), c (float)]
    3.  interchange i-th and j-th row; input: [3, i (int), j (int)]
    """
    def __init__(self, matrix, type: str = 'row', mode: str = 'interactive') -> None:
        self.matrix = np.array(matrix)
        self.m, self.n = self.matrix.shape
        if type not in ['row', 'congruence']:
            raise ValueError("type must be one of ['row', 'congurence']")
        if mode not in ['interactive','independent']:
            raise ValueError("mode must be one of ['interactive','independent']")
        if type == 'congruence' and self.m!=self.n:
            raise ValueError("Congruence operations can only be performed on square matrices.")
        self.type = type
        self.mode = mode

    def operate(self, args: list = None):
        """
        args: (optional) list. Considered only when mode='independent'. Ignored if mode='interactive'.
        Since the operations covered are elementary row and congurence, only row operations need to be taken as input.
        1.  add c times j-th row to i-th row; input: [1, c (float), j (int), i (int)]
        2.  multiply i-th row by c; input: [2, i (int), c (float)]
        3.  interchange i-th and j-th row; input: [3, i (int), j (int)]
        """
        if self.mode == 'interactive':
            if self.type == 'row':
                self.__rowinter__()
            else:
                self.__conginter__()
        elif self.mode == 'independent':
            if self.type == 'row':
                self.__rowind__(args)
            else:
                self.__congind__(args)

    def __rowinter__(self):
        s = ("1. add c times j-th row to i-th row\t input: 1 c j i\n"
            "2. multiply i-th row by c\t\t input: 2 i c\n"
            "3. interchange i-th and j-th rows\t input: 3 i j\n"
            "Input 'end' to stop.\n"
            "(note: replace letters with values.)\n")
        print("Matrix:")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")
        while True:
            var = input(s).split()
            if var[0] == 'end':
                break
            if var[0] == 1:
                _,c,j,i= var
                c = float(c); j = int(i); i = int(i)
                print(f"R'{i} = R{i} + {c}*R{j}")
                self.matrix[i-1] = self.matrix[i-1] + c*self.matrix[j-1]
                print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
            elif var[0] == 2:
                _,i,c = var
                c = float(c); i = int(i)
                print(f"R'{i} = {c}*R{i}")
                self.matrix[i-1] = c*self.matrix[i-1]
                print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
            if var[0] == 3:
                _,i,j = var
                i = int(i); j = int(j)
                print(f"R{i} <-> R{j}")
                self.matrix[[i-1, j-1]] = self.matrix[[j-1, i-1]]
                print('\n'.join(['\t'.join([str(round(cell)) for cell in row]) for row in self.matrix]) + "\n")

    def __conginter__(self):
        s = ("1. add c times j-th row/column to i-th row\t input: 1 c j i\n"
            "2. multiply i-th row/column by c\t\t input: 2 i c\n"
            "3. interchange i-th and j-th rows/columns\t input: 3 i j\n"
            "Input 'end' to stop.\n"
            "(note: replace letters with values.)\n")
        print("Matrix:")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")
        while True:
            var = input(s).split()
            if var[0] == 'end':
                break
            if var[0] == 1:
                _,c,j,i= var
                c = float(c); j = int(i); i = int(i)
                print(f"R'{i} = R{i} + {c}*R{j}\nC'{i} = C{i} + {c}*C{j}")
                self.matrix[i-1] = self.matrix[i-1] + c*self.matrix[j-1]
                self.matrix[:,[i-1]] = self.matrix[:,[i-1]] + c*self.matrix[:,[j-1]]
                print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
            elif var[0] == 2:
                _,i,c = var
                c = float(c); i = int(i)
                print(f"R'{i} = {c}*R{i}\nC'{i} = {c}*C{i}")
                self.matrix[i-1] = c*self.matrix[i-1]
                self.matrix[:,[i-1]] = c*self.matrix[:,[i-1]]
                print('\n'.join(['\t'.join([str(round(cell, 2)) for cell in row]) for row in self.matrix]) + "\n")
            if var[0] == 3:
                _,i,j = var
                i = int(i); j = int(j)
                print(f"R{i} <-> R{j}\nC{i} <-> C{j}")
                self.matrix[[i-1, j-1]] = self.matrix[[j-1, i-1]]
                self.matrix[:, [i-1, j-1]] = self.matrix[:, [j-1, i-1]]
                print('\n'.join(['\t'.join([str(round(cell)) for cell in row]) for row in self.matrix]) + "\n")

    