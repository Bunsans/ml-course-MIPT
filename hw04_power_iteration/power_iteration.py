import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    init_vec = np.random.random(data.shape[0])
    for _ in range(num_steps):
        next_vec = data @ init_vec / np.linalg.norm(data @ init_vec)
        init_vec = next_vec
    eigenvalue = next_vec.T @ (data @ next_vec) / (next_vec.T @ next_vec)

    return float(eigenvalue), next_vec
