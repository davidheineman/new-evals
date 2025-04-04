from tqdm import tqdm
import numpy as np
import json
from scipy import optimize, special
from concurrent.futures import ProcessPoolExecutor


def p_theta(theta, a, b):
    return 1 / (1 + np.exp(-a * (theta - b)))


def fisher_information(theta, a, b):
    p = p_theta(theta, a, b)
    return a**2 * p * (1 - p)


def test_information(theta, a_list, b_list):
    item_information = np.zeros((len(a_list), len(theta)))
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        item_information[i, :] = fisher_information(theta, a, b)
    return item_information.sum(axis=0)


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


def calculate_theta(difficulties, discriminations, responses, func="bernoulli", quiet=False):
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
            desc="Calculating abilities",
            disable=quiet
        ))
    
    return thetas


def save_irt_params(save_path, train_model_names, train_instance_names, discriminations, difficulties, thetas):
    # Save IRT parameters
    params = {
        'a': {name: disc for name, disc in zip(train_instance_names, discriminations)},
        'b': {name: diff for name, diff in zip(train_instance_names, difficulties)},
        'theta': {name: theta for name, theta in zip(train_model_names, thetas)}
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=4)

    return save_path


def load_irt_params(load_path):
    with open(load_path, "r") as f:
        irt_params = json.load(f)

    # Extract IRT params
    items = list(irt_params["a"].keys())
    discriminations = np.array([irt_params["a"][i] for i in items])
    difficulties = np.array([irt_params["b"][i] for i in items])

    return items, discriminations, difficulties


# import huggingface_hub as hf_hub
# IRT_REPO = "allenai/irt-evals"  # https://huggingface.co/datasets/allenai/irt-evals
# IRT_FILEPATH = "{version}/{metric_name}/{task_alias}.json"
# def pull_irt_params(task_alias):
#     """Pull trained IRT parameters from HF"""
#     remote_filename = IRT_FILEPATH.format(
#         version="v0", metric_name="primary_metric", task_alias=task_alias
#     )

#     local_path = hf_hub.hf_hub_download(
#         repo_id=IRT_REPO, filename=remote_filename, repo_type="dataset"
#     )
    
#     items, discriminations, difficulties = load_irt_params(local_path)

#     return items, discriminations, difficulties