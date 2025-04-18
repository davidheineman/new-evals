import numpy as np
import pandas as pd
import warnings
import duckdb

# from db import get_slice_db, get_nd_array_db, get_instance_db
from db import get_slice_db, get_nd_array_db, get_instance_db

def get_slice(df, mix=None, model=None, task=None, step=None, size=None, seed=None):
    """ Index to return a df of some (data mix, model, task, step, seed) """
    if isinstance(df, duckdb.DuckDBPyConnection):
        return get_slice_db(df, mix, model, task, step)
    
    mixes   = [mix] if isinstance(mix, str) else mix
    models  = [model] if isinstance(model, str) else model
    tasks   = [task] if isinstance(task, str) else task
    steps   = [step] if isinstance(step, int) else step
    sizes   = [size] if isinstance(size, str) else size
    seeds   = [seed] if isinstance(seed, (int, np.integer)) else seed

    # Dynamically create a slicing tuple matching the index levels
    level_slices = {
        'mix':    mixes if mixes else slice(None),
        'model':  models if models else slice(None),
        'task':   tasks if tasks else slice(None),
        'step':   steps if steps else slice(None),
        'size':   sizes if sizes else slice(None),
        'seed':   seeds if seeds else slice(None)
    }
    slicing_tuple = tuple(level_slices.get(level, slice(None)) for level in df.index.names)

    is_multiindex = isinstance(df.index, pd.MultiIndex)

    if is_multiindex:
        try:
            df = df.loc[slicing_tuple]
        except KeyError:
            return df.iloc[0:0]  # Return an empty DataFrame if no match
    else:
        # Slow index
        df = df[
            (df['mix'].isin(level_slices['mix'])     if isinstance(level_slices['mix'], list) else True) &
            (df['model'].isin(level_slices['model']) if isinstance(level_slices['model'], list) else True) &
            (df['task'].isin(level_slices['task'])   if isinstance(level_slices['task'], list) else True) &
            (df['step'].isin(level_slices['step'])   if isinstance(level_slices['step'], list) else True) &
            (df['size'].isin(level_slices['size'])   if isinstance(level_slices['size'], list) else True) &
            (df['seed'].isin(level_slices['seed'])   if isinstance(level_slices['seed'], list) else True)
        ]
    
    # Sort and return
    if 'step' in df.index.names:
        df = df.sort_index(level='step')
    df = df.reset_index()

    return df


def get_instance(df, instance_id):
    """ Index to return a df of some (instance_id) """
    if isinstance(df, duckdb.DuckDBPyConnection):
        return get_instance_db(df, instance_id)

    instance_ids   = [instance_id] if isinstance(instance_id, str) else instance_id

    # Dynamically create a slicing tuple matching the index levels
    level_slices = {
        'instance_id': instance_ids if instance_ids else slice(None),
    }
    slicing_tuple = tuple(level_slices.get(level, slice(None)) for level in df.index.names)

    is_multiindex = isinstance(df.index, pd.MultiIndex)

    if is_multiindex:
        try:
            df = df.loc[slicing_tuple]
        except KeyError:
            return df.iloc[0:0]  # Return an empty DataFrame if no match
    else:
        # Slow index
        df = df[
            (df['instance_id'].isin(level_slices['instance_id']) if isinstance(level_slices['instance_id'], list) else True)
        ]
    
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


def get_nd_array(df, col, metric, mix=None, model=None, task=None, step=None, sorted=False, return_index=False):
    """ Get an nd array of (COL, instances), sorted by overall performance """
    if isinstance(df, duckdb.DuckDBPyConnection):
        return get_nd_array_db(df, col, metric, mix, model, task, step, sorted, return_index)

    col = [col] if not isinstance(col, list) else col
    
    use_max_step = False
    if step == 'max':
        use_max_step = True
        step = None
    
    slices = get_slice(df, mix, model, task, step)

    if len(slices) == 0:
        # raise RuntimeError(f'Encountered empty slice: {slices}')
        if return_index:
            return [], [], np.array([])
        return [], np.array([])

    if use_max_step: 
        slices = get_max_k_step(slices)

    is_multiindex = isinstance(df.index, pd.MultiIndex)

    if is_multiindex:
        # For native_ids which count up from 0, there are the same IDs across tasks. Append the task name.
        slices['native_id'] = slices['native_id'] + ':' + slices['task'].astype(str)
        
        duplicates_count = slices.duplicated(subset=['native_id'] + col).sum()
        if duplicates_count > 0:
            if 'hellaswag' not in task and \
                'drop' not in task: # this is a known problem for 433 HellaSwag instances, 1 Drop instance
                warnings.simplefilter("once", UserWarning)
                warnings.warn(f"Warning: {duplicates_count}/{len(slices)} duplicate native_id-key pairs found for task='{task}' model='{model}'. Removing duplicates...", category=UserWarning, stacklevel=2)
            slices = slices.drop_duplicates(subset=['native_id'] + col, keep='first')

        # Pivot the data to get mixes as columns and question_ids as rows
        pivoted = slices.pivot(index='native_id', columns=col, values=metric)

        columns = pivoted.columns
        index = pivoted.index
        scores = pivoted.to_numpy()
    else:
        if len(col) == 1:
            columns = slices[col[0]].to_numpy()
            scores  = slices[metric].to_numpy()
            index   = slices.index
        else:
            pivoted = slices.pivot(index=col[0], columns=col[1:], values=metric)
            index = pivoted.index
            columns = pivoted.columns
            scores = pivoted.to_numpy()

    if is_multiindex:
        # If there are multiple cols, reshape the output nd array
        if len(col) > 1:
            # pivoted = pivoted.sort_index(axis=1)
            # expanded_columns = pivoted.columns.to_frame(index=False)
            # pivoted.columns = pd.MultiIndex.from_tuples(
            #     [tuple(col) for col in expanded_columns.to_numpy()],
            #     names=expanded_columns.columns.tolist()
            # )
            # scores = pivoted.to_numpy()
            # scores = scores.reshape(
            #     (pivoted.shape[0], len(expanded_columns['mix'].unique()), len(expanded_columns['step'].unique()))
            # )
            pivoted = pivoted.sort_index(axis=1)
            expanded_columns = pivoted.columns.to_frame(index=False)
            pivoted.columns = pd.MultiIndex.from_tuples(
                [tuple(col) for col in expanded_columns.to_numpy()],
                names=expanded_columns.columns.tolist()
            )
            scores = pivoted.to_numpy()
            unique_counts = [len(expanded_columns[level].unique()) for level in expanded_columns.columns]
            scores = scores.reshape((pivoted.shape[0], *unique_counts))

        # # Add a new axis for dim=1 if necessary
        # scores = np.expand_dims(scores, axis=1)

    # Move instances dim to final dim
    scores = np.moveaxis(scores, 0, -1)

    if sorted:
        if len(col) == 1 and not is_multiindex: 
            sorted_indices = np.argsort(scores)
            columns = columns[sorted_indices]
            # index = index[sorted_indices]
            scores  = scores[sorted_indices]
        else:
            # Sort by overall performance
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            columns = columns[sorted_indices].tolist()
            # index = index[sorted_indices].tolist()
            scores = scores[sorted_indices]

    if not isinstance(columns, list): 
        columns = columns.tolist()
    if not isinstance(index, list): 
        index = index.tolist()

    if return_index:
        return index, columns, scores
    return columns, scores
