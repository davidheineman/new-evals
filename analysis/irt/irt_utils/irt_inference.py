from tqdm import tqdm
import numpy as np
from scipy import optimize, special
from concurrent.futures import ProcessPoolExecutor


def theta_bernoulli_fn(difficulties, discriminations, responses):
    """Return a function to compute the theta ability parameter for multiple response sets"""
    def fn(theta):
        # Compute probabilities once for all response sets
        probabilities = special.expit(discriminations * (theta - difficulties))
        # Clip probabilities to avoid log errors
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
        
        # Get log probabilities once
        log_prob = np.log(probabilities)
        log_1_minus_prob = np.log(1 - probabilities)
        
        # Compute log likelihood for this response set
        log_likelihood = np.where(responses == 0, log_1_minus_prob, log_prob).sum()
        return -log_likelihood  # Negative for minimization

    return fn


def theta_gaussian_fn(difficulties, discriminations, responses, sigma=0.1):
    """Return a function to compute the theta ability parameter for multiple continuous response sets"""
    def fn(theta):
        probabilities = special.expit(discriminations * (theta - difficulties))
        probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)  # Prevent log errors
        
        # Gaussian likelihood for continuous responses
        log_likelihood = -0.5 * np.log(2 * np.pi * sigma**2) - ((responses - probabilities) ** 2) / (2 * sigma**2)
        
        return -log_likelihood.sum()  # Negative for minimization

    return fn


def _optimize_single(theta_fn, args):
    """Helper function to optimize a single response set"""
    difficulties, discriminations, response_set = args
    fn = theta_fn(difficulties, discriminations, response_set)
    result = optimize.minimize(fn, 0.1, method="BFGS")
    return result["x"].item()


def calculate_theta(difficulties, discriminations, responses, func="bernoulli"):
    """Calculate the ability param for multiple response sets in parallel"""
    # Create args for each optimization
    args = [(difficulties, discriminations, response_set) for response_set in responses]

    if func == "bernoulli":
        theta_fn = theta_bernoulli_fn
    elif func == "gaussian":
        theta_fn = theta_gaussian_fn
    else:
        raise RuntimeError(func)
    
    with ProcessPoolExecutor() as executor:
        thetas = list(tqdm(
            executor.map(_optimize_single, [theta_fn] * len(args), args),
            total=len(args),
            desc="Calculating abilities"
        ))
    
    return thetas
