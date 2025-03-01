import argparse
import json
import random

import numpy as np
import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer
from scipy import stats

from two_param_logistic import Birnbaum


def main():

    # Make reproducible
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    pyro.set_rng_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        required=True
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False
    )
    parser.add_argument(
        "--check_stability", 
        action="store_true",
        required=False
    )
    args = parser.parse_args()

    # Define directories
    data_dir = "/mnt/c/Users/valen/Documents/Research/irt-evals/data"
    params_dir = "/mnt/c/Users/valen/Documents/Research/irt-evals/irt_params"

    # Load input data
    print(f"Processing {args.input_data}...")
    data = Dataset.from_jsonlines(f"{data_dir}/{args.input_data}.jsonlines")
    config = IrtConfig(
        model_type=Birnbaum,
        priors="hierarchical",
    )

    # Stability check
    if args.check_stability:
        best_parameters_data = []
        for i in range(3):
            trainer = IrtModelTrainer(
                config=config, 
                data_path=None, 
                dataset=data
            )
            trainer.train(epochs=1000, device=args.device)
            best_parameters_data.append(trainer.best_params)
        parameters = ["ability", "diff", "disc"]
        for parameter in parameters:
            r_list = []
            for i in range(3):
                for j in range(i + 1, 3):
                    r_list.append(stats.pearsonr(
                        best_parameters_data[i][parameter],
                        best_parameters_data[j][parameter]
                    )[0])
            print(f"Correlation for {parameter}: {np.mean(r_list)}")
        return 0

    # Fit IRT model
    trainer = IrtModelTrainer(
        config=config, 
        data_path=None, 
        dataset=data
    )
    trainer.train(epochs=1000, device=args.device)
    del trainer.best_params["irt_model"]  # Not serializable
    with open(f"{params_dir}/{args.input_data}.json", "w") as f:
        json.dump(trainer.best_params, f, indent=4)


if __name__ == "__main__":
    main()
