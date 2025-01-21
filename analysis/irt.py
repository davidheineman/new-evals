from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np

def itc_filtering(step_scores, percentile=50):
    # Compute item-total correlations
    total_scores = step_scores.mean(axis=1)  # Average across instances for each model
    item_total_corrs = np.array([
        pearsonr(step_scores[:, i], total_scores)[0]
        for i in range(step_scores.shape[1])
    ])

    # Filter out lowest correlating instances (keep top 50%)
    # Handle case where all correlations are NaN/invalid
    if np.all(np.isnan(item_total_corrs)):
        # Keep all instances if correlations are invalid
        high_corr_mask = np.ones_like(item_total_corrs, dtype=bool)
        filtered_scores = step_scores
        threshold = 0
    else:
        threshold = np.percentile(item_total_corrs[~np.isnan(item_total_corrs)], percentile)
        high_corr_mask = item_total_corrs >= threshold
        filtered_scores = step_scores[:, high_corr_mask]

    return filtered_scores


def variance_filtering(step_scores, percentile=50):
    # Compute variance across models for each instance
    instance_variances = np.var(step_scores, axis=0)
    
    # Handle case where all variances are NaN/invalid
    if np.all(np.isnan(instance_variances)):
        # Keep all instances if variances are invalid
        high_var_mask = np.ones_like(instance_variances, dtype=bool)
        filtered_scores = step_scores
        threshold = 0
    else:
        # Filter out lowest variance instances (keep top percentile%)
        valid_variances = instance_variances[~np.isnan(instance_variances)]
        k = int(np.ceil(len(valid_variances) * (100 - percentile) / 100))  # Number to keep
        threshold = np.partition(valid_variances, -k)[-k]  # kth largest value
        high_var_mask = instance_variances >= threshold
        # Ensure we keep exactly percentile% of valid instances
        if np.sum(high_var_mask) > k:
            # If there are ties at threshold, randomly select among them
            tied = instance_variances == threshold
            n_excess = np.sum(high_var_mask) - k
            tied_indices = np.where(tied)[0]
            to_remove = np.random.choice(tied_indices, size=n_excess, replace=False)
            high_var_mask[to_remove] = False
        filtered_scores = step_scores[:, high_var_mask]

    return filtered_scores


def bradley_terry_scores(train_scores, test_scores=None):
    """ Compute a Bradley Terry ranking """
    num_train = train_scores.shape[0]
    
    # Initialize ratings randomly between 0 and 1
    logits = np.zeros(num_train)
    
    # Iterative fitting (maximum 100 iterations)
    for _ in tqdm(range(100), desc='Fitting BT model'):
        old_logits = logits.copy()
        
        # For each model
        for i in range(num_train):
            # Compare against each other model
            wins = 0
            total = 0
            for j in range(num_train):
                if i != j:
                    # Count instances where model i beats model j
                    wins += np.sum(train_scores[i] > train_scores[j])
                    # Count total comparable instances
                    total += np.sum((train_scores[i] != -np.inf) & (train_scores[j] != -np.inf))
            
            # Skip if no comparisons
            if total == 0:
                continue
                
            # Compute expected wins based on current ratings
            exp_logits = np.exp(logits - logits[i])  # Use relative logits
            probs = 1 / (1 + exp_logits)
            expected = np.sum(probs[np.arange(num_train) != i])
            
            # Update logit
            if wins > 0:
                logits[i] += np.log(wins / expected)
            
        # Check convergence
        if np.max(np.abs(logits - old_logits)) < 1e-4:
            break
            
    # Convert training logits to probabilities
    exp_logits = np.exp(logits - np.max(logits))
    train_ratings = exp_logits / np.sum(exp_logits)
    
    if test_scores is not None:
        # For each test model, compute wins against training models
        num_test = test_scores.shape[0]
        test_logits = np.zeros(num_test)
        
        for i in tqdm(range(num_test), desc='Computing test scores'):
            wins = 0
            total = 0
            for j in range(num_train):
                wins += np.sum(test_scores[i] > train_scores[j])
                total += np.sum((test_scores[i] != -np.inf) & (train_scores[j] != -np.inf))
            
            if total > 0:
                win_ratio = wins / total
                # Convert win ratio to logit relative to training models
                test_logits[i] = np.log(win_ratio / (1 - win_ratio)) if win_ratio < 1 else logits.max()
        
        # Convert test logits to probabilities
        exp_logits = np.exp(test_logits - np.max(test_logits))
        test_ratings = exp_logits / np.sum(exp_logits)
        
        return train_ratings, test_ratings
        
    return train_ratings, None