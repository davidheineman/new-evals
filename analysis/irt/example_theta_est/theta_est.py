import numpy as np
from scipy import optimize, special


def theta_fn(difficulties, discriminations, responses):
    def fn(theta):
        probabilities = special.expit(
            discriminations * (theta - difficulties)
        )
        # Clip probabilities to avoid log errors
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
        log_likelihood = 0
        for i, response in enumerate(responses):
            if response == 0:
                log_likelihood += np.log(1 - probabilities[i])
            else:
                log_likelihood += np.log(probabilities[i])
        return -log_likelihood  # Negative for minimization

    return fn


def calculate_theta(difficulties, discriminations, responses):
    fn = theta_fn(difficulties, discriminations, responses)
    result = optimize.minimize(fn, 0.1, method="BFGS")
    return result["x"].item()
