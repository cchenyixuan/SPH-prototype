import numpy as np


def solve_2d_poisson(source_matrix, domain_range, distance, num_samples):
    # Step 1: Set up the problem domain and discretize it
    x_min, x_max, y_min, y_max = domain_range
    dx = dy = distance
    nx, ny = num_samples, num_samples

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    # Step 2: Define the Laplacian operator matrix using finite difference method
    Lx = np.diag(-2 * np.ones(nx - 1)) + np.diag(np.ones(nx - 2), 1) + np.diag(np.ones(nx - 2), -1)
    Ly = np.diag(-2 * np.ones(ny - 1)) + np.diag(np.ones(ny - 2), 1) + np.diag(np.ones(ny - 2), -1)
    Ix = np.eye(nx - 1)
    Iy = np.eye(ny - 1)
    Laplacian = np.kron(Iy, Lx) / (dx ** 2) + np.kron(Ly, Ix) / (dy ** 2)

    # Step 3: Solve the eigenvalue problem to diagonalize the Laplacian operator matrix
    eigenvalues, eigenmodes = np.linalg.eig(Laplacian)

    # Step 4: Compute the Fourier coefficients of the source term using the eigenmodes
    source_coeffs = np.dot(eigenmodes.T, source_matrix.flatten())

    # Step 5: Calculate the solution in Fourier space
    eigenvalues = eigenvalues.reshape((ny - 1, nx - 1))
    solution_coeffs = source_coeffs / (eigenvalues ** 2)

    # Step 6: Transform the solution back to real space
    solution = np.dot(eigenmodes, solution_coeffs).reshape((ny - 1, nx - 1))

    return solution


# Test the function with a sample source matrix
if __name__ == "__main__":
    domain_range = (-2.50125, 2.50125, -2.50125, 2.50125)
    distance = 0.025
    num_samples = 201

    # Sample source matrix (you can replace this with your own input)
    source_matrix = np.zeros((num_samples, num_samples))
    source_matrix[num_samples // 2, num_samples // 2] = 1.0  # Single point source in the center

    # Call the solver function
    solution = solve_2d_poisson(source_matrix, domain_range, distance, num_samples)

    # Print the solution
    print(solution)
