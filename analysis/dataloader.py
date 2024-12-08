import numpy as np

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
    
    slices = get_slice(df, mix, model, task, step)

    if len(slices) == 0:
        # raise RuntimeError(f'Encountered empty slice: {slices}')
        return [], np.array([])

    if step == 'max': 
        slices = get_max_k_step(slices)
    # elif step <= 10:
    #     slices = get_max_k_step(slices, step)
    #     assert len(set(slices['step'].unique())) == step, f'Did not get the requested number of steps: {step}. Do they exist in the df?'
    
    # Pivot the data to get mixes as columns and question_ids as rows
    pivoted = slices.pivot(index='native_id', columns=col, values=metric)
    columns = pivoted.columns
    scores = pivoted.to_numpy()
        
    # Create a new axis for each provided column
    if len(col) > 1:
        dims = [len(pivoted)]
        for pivot_col in columns.levels:
            dims += [max(1, len(pivot_col))]
        scores = scores.reshape(*dims)

    # Move instances dim to final dim
    scores = np.moveaxis(scores, 0, -1)

    if sorted:
        # Sort by overall performance
        mix_sums = scores.sum(axis=1)
        sorted_indices = mix_sums.argsort()[::-1]
        columns = columns[sorted_indices].tolist()
        scores = scores[sorted_indices]

    return columns, scores
