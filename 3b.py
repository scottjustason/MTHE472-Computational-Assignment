import numpy as np
import matplotlib.pyplot as plt

# System matrices
A = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 2, 1], [0, 0, 0, 4]])
C = np.array([[2, 0, 0, 0]])

# Check the eigenvalues of A for stability
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of A:", eigenvalues)
if np.any(np.abs(eigenvalues) > 1):
    print("The system is unstable: some eigenvalues have an absolute value greater than 1.")

# Process and measurement noise covariance matrices (identity matrices for simplicity)
Q = np.eye(A.shape[0])
R = np.eye(C.shape[0])

# Number of time steps
T = 1000

# Initialize state and observation sequences
x = np.zeros((A.shape[0], T))
y = np.zeros((C.shape[0], T))
x_hat = np.zeros((A.shape[0], T))  # Estimate of x
P = np.eye(A.shape[0])  # Initial error covariance

# Simulate the system
for t in range(1, T):
    try:
        # Simulate the process dynamics
        x[:, t] = A @ x[:, t-1] + np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        y[:, t] = C @ x[:, t] + np.random.multivariate_normal(np.zeros(R.shape[0]), R)

        # Kalman Filter Predict
        x_hat_pred = A @ x_hat[:, t-1]
        P_pred = A @ P @ A.T + Q

        # Kalman Filter Update
        y_residual = y[:, t] - C @ x_hat_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        x_hat[:, t] = x_hat_pred + K @ y_residual
        P = (np.eye(A.shape[0]) - K @ C) @ P_pred

        # Check for numerical stability
        if not np.all(np.isfinite(x[:, t])):
            raise RuntimeError("Numerical instability detected: non-finite state values.")
    except RuntimeError as e:
        print(e)
        break

# Plot the results
plt.figure(figsize=(12, 9))

# Plot true state x_t
plt.plot(x[0, :], label='True state $x_1$', color='blue')

# Plot state estimates \hat{x}_t
plt.plot(x_hat[0, :], label='Estimated state $\hat{x}_1$', color='red')

# Difference between true state and estimate
plt.plot(x[0, :] - x_hat[0, :], label='Estimation error $x_1 - \hat{x}_1$', color='green')

plt.xlabel('Time step')
plt.ylabel('Value')
plt.title('Kalman Filter Simulation')
plt.legend()
plt.show()
