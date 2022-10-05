import numpy as np

class KalmanFilter:
    """
    Simple Kalman filter
    """

    def __init__(self, X, A, R, Z, C, Q, P, B=np.array([0]), U=np.array([0])):
        """
        Initialise the filter
        Args:
            X: State estimate
            R: Estimate covariance
            A: State transition model
            B: Control matrix
            U: Control vector
            R: Process noise covariance
            Z: Measurement of the state X
            C: Observation model
            Q: Observation noise covariance
        """
        self.X = X
        self.P = P
        self.A = A
        self.B = B
        self.U = U
        self.R = R
        self.Z = Z
        self.C = C
        self.Q = Q

    def predict(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.U: Control vector
        Returns:
            updated self.X
        """
        # Project the state ahead
        self.X = self.A @ self.X + self.B @ self.U
        self.P = self.A @ self.P @ self.A.T + self.R

        return self.X

    def update(self, Z):
        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
        """
        K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Q)
        self.X += K @ (Z - self.C @ self.X)
        self.P = self.P - K @ self.C @ self.P

        return self.X

def predict_and_update(coordinates,kalman_filters):
    """
    Perform Kalman predict and updated
    Args:
        coordinates: 2x4 dim matrix with coordinates snout,l_ear,r_ear,tail
        kalman_filters: 1x4 dim vector with kalman filters in same order as above
    Returns:

    """
    predictions = np.zeros(coordinates.shape)
    for i, kalman in enumerate(kalman_filters):
        x = coordinates[0,i]
        y = coordinates[1,i]
        current_measurement = np.array([[x], [y]])
        current_prediction = kalman.predict()
        kalman.update(current_measurement)
        predictions[0,i] = current_prediction[0]
        predictions[1,i] = current_prediction[1]

    return predictions



def create_kalman_filters(start_coordinates,T=1/10):
    """
    Create kalman filters for snout,l_ear,r_ear,tail

    Returns:
        Initialized KalmanFilter objects
    """
    state_matrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
    estimate_covariance = np.eye(state_matrix.shape[0])

    transition_matrix = np.array([[1, 0, T, 0],[0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    process_noise_cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
    measurement_state_matrix = np.zeros((2, 1), np.float32)
    observation_matrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    measurement_noise_cov = np.array([[1,0],[0,1]], np.float32) * 1
    kalman_filters = []
    for i,mouse_part in enumerate(start_coordinates.T):
        state_matrix[0] = mouse_part[0]
        state_matrix[1] = mouse_part[1]
        kalman = KalmanFilter(X=state_matrix,
                              P=estimate_covariance,
                              A=transition_matrix,
                              R=process_noise_cov,
                              Z=measurement_state_matrix,
                              C=observation_matrix,
                              Q=measurement_noise_cov)
                              
        kalman_filters.append(kalman)

    return kalman_filters
