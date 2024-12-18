import numpy as np
import pandas as pd
import warnings

def get_slice(df, mix=None, model=None, task=None, step=None):
    """ Index to return a df of some (data mix, model, task, step) """
    mixes   = [mix] if isinstance(mix, str) else mix
    models  = [model] if isinstance(model, str) else model
    tasks   = [task] if isinstance(task, str) else task
    steps   = [step] if isinstance(step, int) else step

    # Dynamically create a slicing tuple matching the index levels
    level_slices = {
        'mix':    mixes if mixes else slice(None),
        'model':  models if models else slice(None),
        'task':   tasks if tasks else slice(None),
        'step':   steps if steps else slice(None)
    }
    slicing_tuple = tuple(level_slices.get(level, slice(None)) for level in df.index.names)

    try:
        df = df.loc[slicing_tuple]
    except KeyError:
        return df.iloc[0:0]  # Return an empty DataFrame if no match
    
    # Sort and return
    if 'step' in df.index.names:
        df = df.sort_index(level='step')
    df = df.reset_index()

    return df


def get_max_k_step(_slice, k=1):
    """Filter for only rows with the top 5 steps."""
    top_steps = _slice['step'].nlargest(k).unique()
    step_filter = _slice['step'].isin(top_steps)
    _slice = _slice[step_filter]
    return _slice


def get_nd_array(df, col, metric, mix=None, model=None, task=None, step=None, sorted=False):
    """ Get an nd array of (COL, instances), sorted by overall performance """
    col = [col] if not isinstance(col, list) else col
    
    use_max_step = False
    if step == 'max':
        use_max_step = True
        step = None
    
    slices = get_slice(df, mix, model, task, step)

    if len(slices) == 0:
        # raise RuntimeError(f'Encountered empty slice: {slices}')
        return [], np.array([])

    if use_max_step: 
        slices = get_max_k_step(slices)

    # For native_ids which count up from 0, there are the same IDs across tasks. Append the task name.
    slices['native_id'] = slices['native_id'] + '_' + slices['task'].astype(str)
    
    duplicates_count = slices.duplicated(subset=['native_id'] + col).sum()
    if duplicates_count > 0:
        if 'hellaswag' not in task and \
            'drop' not in task: # this is a known problem for 433 HellaSwag instances, 1 Drop instance
            warnings.simplefilter("once", UserWarning)
            warnings.warn(f"Warning: {duplicates_count}/{len(slices)} duplicate native_id-key pairs found for task='{task}'. Removing duplicates...", category=UserWarning, stacklevel=2)
        slices = slices.drop_duplicates(subset=['native_id'] + col, keep='first')

    # Pivot the data to get mixes as columns and question_ids as rows
    pivoted = slices.pivot(index='native_id', columns=col, values=metric)
    columns = pivoted.columns
    scores = pivoted.to_numpy()

    # If there are multiple cols, reshape the output nd array
    if len(col) > 1:
        pivoted = pivoted.sort_index(axis=1)
        expanded_columns = pivoted.columns.to_frame(index=False)
        pivoted.columns = pd.MultiIndex.from_tuples(
            [tuple(col) for col in expanded_columns.to_numpy()],
            names=expanded_columns.columns.tolist()
        )
        scores = pivoted.to_numpy()
        scores = scores.reshape(
            (pivoted.shape[0], len(expanded_columns['mix'].unique()), len(expanded_columns['step'].unique()))
        )

    # # Add a new axis for dim=1 if necessary
    # scores = np.expand_dims(scores, axis=1)

    # Move instances dim to final dim
    scores = np.moveaxis(scores, 0, -1)

    if sorted:
        if len(col) > 1: raise NotImplementedError()
        # Sort by overall performance
        mix_sums = scores.sum(axis=1)
        sorted_indices = mix_sums.argsort()[::-1]
        columns = columns[sorted_indices].tolist()
        scores = scores[sorted_indices]

    return columns, scores
