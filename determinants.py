import numpy as np


class Determinant(object):
    """
    Calculates the determinant of a matrix.

    Parameters:
    ----------
    matrix: list of lists, that is a matrix of dimension (n,n). Can or cannot be a numpy array.
    """

    def __init__(self, matrix: np.array) -> None:
        self.matrix = np.array(matrix)

    def calc_det(self):
    
        n = len(self.matrix)
        assert self.matrix.shape == (n,n), "Only square matrices allowed"
        print("Matrix:")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")

        for col in range(0, n + 1):
            for row in range(col + 1, n):

                if self.matrix[row][col] != 0:
                    print(f"R'{row+1} = R{row+1} - {self.matrix[row][col]}/{self.matrix[col][col]}*R{col+1}")
                    self.matrix[row] = self.matrix[row] - (self.matrix[row][col]/self.matrix[col][col])*self.matrix[col]
                    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix]) + "\n")

        det = self.matrix.diagonal().prod()
        print(f"Determinant: {det}")
