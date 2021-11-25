import numpy as np
from matplotlib import pyplot as plt


class MatrixCompletion:
    """
    Soit A une matrice clairsemée de dimensions (p, q).
    On suppose que A soit de rang faible.

    Soit M une matrice de dimensions (p, q) dite masque.
    M(i,j) = 1 ssi A(i, j) est défini, 0 sinon.
    """
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols

        self.M = self.generate_mask(rows, cols)
        self.A = self.generate_A(rows, cols)
        self.X = self.generate_X(rows, cols)

    def generate_mask(self, rows, cols):
        self.M = np.random.choice([0, 1], (rows, cols), p=[0.7, 0.3])
        return self.M

    def generate_X(self, rows, cols):
        self.X = np.random.randint(0, 5, (rows, cols))
        return self.X

    def generate_A(self, rows, cols):
        # A_temp = np.random.random((rows, cols))
        # A_temp = A_temp @ A_temp.T
        self.A = np.random.randint(0, 5, (rows, cols)) * self.M
        # self.A = A_temp * self.M
        return self.A

    def compute_solution(self, x_sol):
        return 1/2 * np.linalg.norm(self.M * (self.A - x_sol))

    def gradient_descent(self, max_iterations, gradient_step, epsilon):
        x_t = self.X
        differences = []

        for it in range(max_iterations):
            x_temp = x_t + gradient_step * (self.M * (self.A - x_t))

            # décomposition SVD
            evaluations, diagonal, users = np.linalg.svd(x_temp)

            # les valeurs singulières sont triées par ordre décroissant
            diagonal_opt = np.diag(diagonal[:self.cols])
            diagonal_opt.resize(self.A.shape)

            x_t = evaluations @ diagonal_opt @ users

            current_distance = self.compute_solution(x_temp)
            differences.append(current_distance)

            if current_distance <= epsilon:
                print('Current distance {}'.format(current_distance))
                print('Converged earlier at iteration {}'.format(it))
                return x_t

        return x_t

    def solve(self, max_iterations=100_000, gradient_step=0.001, epsilon=0.01):
        self.X = self.gradient_descent(max_iterations, gradient_step, epsilon)


if __name__ == "__main__":
    num_users = 100
    num_evaluations = 7

    solver = MatrixCompletion(num_users, num_evaluations)

    #print(solver.A)
    #print('-' * 20)
    solver.solve()
    #print(solver.X * solver.M)

    print(np.linalg.norm(solver.A - solver.X * solver.M))
