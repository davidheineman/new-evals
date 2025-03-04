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
from py_irt.models.abstract_model import IrtModel
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
    # Make reproducible
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    pyro.set_rng_seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

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
        train_items=predictions,
        train_model_names=model_names,
        instance_names=instance_names,
    )
    
    trainer = IrtModelTrainer(config=config, dataset=_dataset, data_path=None)
    console.log("Training Model...")
    trainer.train(device=device)
    # params = trainer.last_params
    params = trainer.best_params
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Train time:", elapsed_time)

    return trainer.irt_model, params


def evaluate(
    irt_model,
    predictions: np.ndarray,
    model_names: str,
    instance_names: str,
    quiet: bool = True,
):
    start_time = time.time()

    if quiet:
        console.quiet = True

    _dataset = Dataset.from_ndarray(
        test_items=predictions,
        test_model_names=model_names,
        instance_names=instance_names,
    )

    # get validation data
    # filter out test data
    console.log("Evaluating Model...")
    testing_idx = [
        i for i in range(len(_dataset.training_example)) if not _dataset.training_example[i]
    ]
    if len(testing_idx) > 0:
        _dataset.observation_subjects = [
            _dataset.observation_subjects[i] for i in testing_idx]
        _dataset.observation_items = [
            _dataset.observation_items[i] for i in testing_idx]
        _dataset.observations = [_dataset.observations[i] for i in testing_idx]
        _dataset.training_example = [
            _dataset.training_example[i] for i in testing_idx]

    preds = irt_model.predict(
        _dataset.observation_subjects, 
        _dataset.observation_items
    )

    outputs = []
    for i in range(len(preds)):
        outputs.append(
            {
                "subject_id": _dataset.observation_subjects[i],
                "example_id": _dataset.observation_items[i],
                "response": _dataset.observations[i],
                "prediction": preds[i],
            }
        )

    import pandas as pd    
    df = pd.DataFrame(outputs)
    results = df[['subject_id', 'prediction']]\
        .set_index('subject_id').to_dict()['prediction']
    # map back fo the original model name
    results = {
        _dataset.ix_to_subject_id[subject_id]: prediction
        for subject_id, prediction in results.items()
    }

    console.log("Evaluation time:", time.time() - start_time)

    return results


def train_and_evaluate(
    model_type: str,
    train_predictions: np.ndarray,
    test_predictions: np.ndarray,
    train_model_names: List[str],
    test_model_names: List[str],
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
    start_time = time.time()

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
    config = load_config(config_path, model_type, args_config)

    _dataset = Dataset.from_ndarray(
        train_items=train_predictions,
        test_items=test_predictions,
        train_model_names=train_model_names,
        test_model_names=test_model_names,
        instance_names=instance_names,
    )

    # deep copy for training
    import copy
    training_data = copy.deepcopy(_dataset)
    trainer = IrtModelTrainer(config=config, dataset=training_data, data_path=None)
    console.log("Training Model...")
    trainer.train(device=device)
    params = trainer.last_params

    # get validation data
    # filter out test data
    console.log("Evaluating Model...")
    testing_idx = [
        i for i in range(len(_dataset.training_example)) if not _dataset.training_example[i]
    ]
    print('hi')
    print(len(testing_idx))
    if len(testing_idx) > 0:
        _dataset.observation_subjects = [_dataset.observation_subjects[i] for i in testing_idx]
        _dataset.observation_items = [_dataset.observation_items[i] for i in testing_idx]
        _dataset.observations = [_dataset.observations[i] for i in testing_idx]
        _dataset.training_example = [_dataset.training_example[i] for i in testing_idx]

    preds = trainer.irt_model.predict(
        _dataset.observation_subjects, 
        _dataset.observation_items
    )

    outputs = []
    for i in range(len(preds)):
        outputs.append(
            {
                "subject_id": _dataset.observation_subjects[i],
                "example_id": _dataset.observation_items[i],
                "response": _dataset.observations[i],
                "prediction": preds[i],
            }
        )

    import pandas as pd    
    df = pd.DataFrame(outputs)
    results = df[['subject_id', 'prediction']]\
        .set_index('subject_id').to_dict()['prediction']
    # map back fo the original model name
    results = {
        _dataset.ix_to_subject_id[subject_id]: prediction
        for subject_id, prediction in results.items()
    }

    console.log("Evaluation time:", time.time() - start_time)

    return (trainer.irt_model, params), results