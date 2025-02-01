import random
from typing import Optional, List, Dict, Any
import time

import torch
import pyro
import numpy as np
import typer
import toml
from rich.console import Console

from py_irt.training import IrtModelTrainer
from py_irt.config import IrtConfig
from py_irt.io import read_json
from .irt_dataset import Dataset

console = Console()
app = typer.Typer()


def setup_random_seeds(seed: Optional[int], deterministic: bool) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)


def load_config(config_path: Optional[str], model_type: str, args_config: Dict[str, Any]) -> IrtConfig:
    """Load and merge configuration from file and arguments."""
    if config_path is None:
        parsed_config = {}
    else:
        if config_path.endswith(".toml"):
            with open(config_path) as f:
                parsed_config = toml.load(f)
        else:
            parsed_config = read_json(config_path)

    # Update config with non-None argument values
    parsed_config.update({k: v for k, v in args_config.items() if v is not None})

    if model_type != parsed_config["model_type"]:
        raise ValueError("Mismatching model types in args and config")

    return IrtConfig(**parsed_config)

def train(
    model_type: str,
    predictions: np.ndarray,
    model_names: str,
    instance_names: str,
    epochs: Optional[int] = None,
    priors: Optional[str] = None,
    dims: Optional[int] = None,
    lr: Optional[float] = None,
    lr_decay: Optional[float] = None,
    device: str = "cpu",
    initializers: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    dropout: Optional[float] = 0.5,
    hidden: Optional[int] = 100,
    seed: Optional[int] = None,
    deterministic: bool = False,
    log_every: int = 100,
    quiet: bool = False,
):
    if quiet:
        console.quiet = True

    args_config = {
        "priors": priors,
        "dims": dims,
        "lr": lr,
        "lr_decay": lr_decay,
        "epochs": epochs,
        "initializers": initializers,
        "model_type": model_type,
        "dropout": dropout,
        "hidden": hidden,
        "log_every": log_every,
        "deterministic": deterministic,
        "seed": seed,
    }

    setup_random_seeds(seed, deterministic)
    config = load_config(config_path, model_type, args_config)
    console.log(f"config: {config}")

    start_time = time.time()
    
    _dataset = Dataset.from_ndarray(
        predictions=predictions,
        model_names=model_names,
        instance_names=instance_names,
    )
    
    trainer = IrtModelTrainer(config=config, dataset=_dataset, data_path=None)
    console.log("Training Model...")
    trainer.train(device=device)
    params = trainer.last_params
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Train time:", elapsed_time)

    return params