import os
import argparse
from comet_ml import API
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

def fetch_experiments_by_run_name(
    api_key: str,
    workspace: str,
    project_name: str,
    run_name: str
) -> List[Dict[str, List[float]]]:
    """
    Fetch all experiments with a specific run name and their downstream metrics

    Args:
        api_key: Comet ML API key
        workspace: Comet ML workspace name
        project_name: Project name
        run_name: Name of the run to search for

    Returns:
        List of dictionaries containing metrics for each matching experiment
    """
    # Initialize Comet ML API
    api = API(api_key=api_key)

    # Get all experiments in the project
    experiments = api.get(workspace=workspace, project_name=project_name)

    # Filter experiments by run name
    matching_experiments = []

    i = 0
    for exp in tqdm(experiments, desc="Fetching experiments", position=0):
        if exp.get_name() != run_name:
            continue

        downstream_metrics_x: Dict[str, List[int]] = {}
        downstream_metrics_y: Dict[str, List[float]] = {}

        for item in tqdm(exp.get_metrics(), desc=f"Fetching metrics for {exp.id}", position=1):
            if item['step'] is None:
                continue
            if not item['metricName'].startswith('eval/downstream'):
                continue

            downstream_metrics_x.setdefault(item['metricName'], []).append(int(item['step']))
            downstream_metrics_y.setdefault(item['metricName'], []).append(float(item['metricValue']))

        if downstream_metrics_y:  # Only include if it has downstream metrics
            matching_experiments.append({
                'id': exp.id,
                'step': downstream_metrics_x,
                'metrics': downstream_metrics_y
            })

    if not matching_experiments:
        print(f"No experiments found with run name: {run_name}")
    else:
        print(f"Found {len(matching_experiments)} experiments with run name: {run_name}")

    return matching_experiments


def plot_metric_comparison(
    metric_name: str,
    experiments: List[Dict],
    output_path: Path
):
    """
    Create visualization comparing a single metric across multiple experiments

    Args:
        metric_name: Name of the metric to plot
        experiments: List of experiment dictionaries containing metrics
        output_path: Path to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot line for each experiment
    legend_labels = []
    for exp in experiments:
        if metric_name in exp['metrics']:
            df = pd.DataFrame({'step': exp['step'][metric_name], 'value': exp['metrics'][metric_name]})

            # Plot line
            sns.lineplot(data=df, x='step', y='value', alpha=0.7)
            legend_labels.append(f"Experiment {exp['id'][:8]}")

    # Customize plot
    clean_name = metric_name.replace('eval/downstream/', '')
    plt.title(clean_name, fontsize=14, pad=20)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistics annotation
    all_values = []
    for exp in experiments:
        if metric_name in exp['metrics']:
            all_values.extend(exp['metrics'][metric_name])


    if all_values:
        stats_text = f"Overall Statistics:\n"
        stats_text += f"Mean: {pd.Series(all_values).mean():.4f}\n"
        stats_text += f"Std: {pd.Series(all_values).std():.4f}\n"
        stats_text += f"Min: {min(all_values):.4f}\n"
        stats_text += f"Max: {max(all_values):.4f}"

        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout to accommodate legend
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Close the figure to free memory
    plt.close()

def plot_all_metrics(experiments: List[Dict], output_dir: Path):
    """
    Create visualizations for all metrics across experiments

    Args:
        experiments: List of experiment dictionaries containing metrics
        output_dir: Directory to save the plots
    """

    # Get unique metric names across all experiments
    metric_names = set()
    for exp in experiments:
        metric_names.update(exp['metrics'].keys())

    # Plot each metric
    for metric_name in sorted(metric_names):
        # Create clean filename from metric name
        clean_name = metric_name.replace('eval/downstream/', '')\
                               .replace('/', '_')\
                               .replace(' ', '_')\
                               .lower()
        output_path = output_dir / f"{clean_name}.png"

        # Plot and save individual metric comparison
        plot_metric_comparison(metric_name, experiments, output_path)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Fetch and visualize Comet ML downstream metrics'
    )

    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help='Comet ML workspace name'
    )

    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Comet ML project name'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Name of the run to analyze'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save the output plots'
    )

    return parser.parse_args()


def main():
    """
    Main function to fetch and visualize metrics
    """
    # Parse arguments
    args = parse_arguments()

    # Get API key from environment
    api_key = os.getenv('COMET_API_KEY')
    if not api_key:
        raise ValueError(
            "Please set the COMET_API_KEY environment variable"
        )

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch experiments and their metrics
    experiments = fetch_experiments_by_run_name(
        api_key=api_key,
        workspace=args.workspace,
        project_name=args.project,
        run_name=args.run_name
    )

    if not experiments:
        return

    # Plot metrics
    plot_all_metrics(experiments, output_dir)
    print(f"\nAll plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()
