import numpy as np
import pandas as pd
import warnings
import clickhouse_connect

def load_db_backend(local_path):
    try:
        client = clickhouse_connect.get_client(host='localhost', port=8123)
        client.query('SELECT 1')
    except Exception:
        raise RuntimeError("clickhouse is not running, please make sure it is installed and has a backend running (`clickhouse server --daemon`)")
    client = clickhouse_connect.get_client(host='localhost', port=8123)
    # Create table from parquet file
    client.command(f"""
        CREATE TABLE IF NOT EXISTS instances
        ENGINE = MergeTree()
        ORDER BY tuple()
        AS SELECT * FROM file('{local_path}', Parquet)
    """)
    return client

def get_slice_db(db, mix=None, model=None, task=None, step=None):
    """ Index to return a df of some (data mix, model, task, step) """
    # Build SQL query conditions
    conditions = []
    if mix:
        mixes = [mix] if isinstance(mix, str) else mix
        conditions.append("mix IN " + (str(tuple(mixes)) if len(mixes) > 1 else f"('{mixes[0]}')"))
    if model:
        models = [model] if isinstance(model, str) else model
        conditions.append("model IN " + (str(tuple(models)) if len(models) > 1 else f"('{models[0]}')"))
    if task:
        tasks = [task] if isinstance(task, str) else task
        conditions.append("task IN " + (str(tuple(tasks)) if len(tasks) > 1 else f"('{tasks[0]}')"))
    if step is not None:
        steps = [step] if isinstance(step, int) else step
        conditions.append("step IN " + (str(tuple(steps)) if len(steps) > 1 else f"({steps[0]})"))

    # Build and execute query
    query = "SELECT * FROM instances"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    # Check if step column exists
    has_step = db.query("SELECT name FROM system.columns WHERE table='instances' AND name='step'").result_rows
    if has_step:
        query += " ORDER BY step"

    df = pd.DataFrame(db.query(query).result_rows, columns=db.query(query).column_names)

    # Re-arrange first cols
    first_cols = ['task', 'model', 'step', 'mix', 'primary_score', 'logits_per_byte_corr']
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    return df

def get_instance_db(db, instance_id):
    """ Index to return a df of some (instance_id) """
    instance_ids = [instance_id] if isinstance(instance_id, str) else instance_id
    
    query = "SELECT * FROM instances WHERE instance_id IN "
    query += str(tuple(instance_ids)) if len(instance_ids) > 1 else f"('{instance_ids[0]}')"
    
    # Check if step column exists
    has_step = db.query("SELECT name FROM system.columns WHERE table='instances' AND name='step'").result_rows
    if has_step:
        query += " ORDER BY step"

    df = pd.DataFrame(db.query(query).result_rows, columns=db.query(query).column_names)
    return df

def get_max_k_step_db(_slice, k=1):
    """Filter for only rows with the top k steps."""
    top_steps = _slice['step'].nlargest(k).unique()
    step_filter = _slice['step'].isin(top_steps)
    _slice = _slice[step_filter]
    return _slice

def get_nd_array_db(db, col, metric, mix=None, model=None, task=None, step=None, sorted=False, return_index=False):
    """ Get an nd array of (COL, instances), sorted by overall performance """
    col = [col] if not isinstance(col, list) else col
    
    use_max_step = False
    if step == 'max':
        use_max_step = True
        step = None
    
    slices = get_slice_db(db, mix, model, task, step)

    if len(slices) == 0:
        if return_index:
            return [], [], np.array([])
        return [], np.array([])

    if use_max_step:
        slices = get_max_k_step_db(slices)

    # Add task name to native_id to ensure uniqueness
    slices['native_id'] = slices['native_id'] + ':' + slices['task'].astype(str)
    
    duplicates_count = slices.duplicated(subset=['native_id'] + col).sum()
    if duplicates_count > 0:
        if 'hellaswag' not in task and 'drop' not in task:
            warnings.simplefilter("once", UserWarning)
            warnings.warn(f"Warning: {duplicates_count}/{len(slices)} duplicate native_id-key pairs found for task='{task}' model='{model}'. Removing duplicates...", category=UserWarning, stacklevel=2)
        slices = slices.drop_duplicates(subset=['native_id'] + col, keep='first')

    # Pivot the data
    pivoted = slices.pivot(index='native_id', columns=col, values=metric)

    columns = pivoted.columns
    index = pivoted.index
    scores = pivoted.to_numpy()

    # Handle multiple columns case
    if len(col) > 1:
        pivoted = pivoted.sort_index(axis=1)
        expanded_columns = pivoted.columns.to_frame(index=False)
        pivoted.columns = pd.MultiIndex.from_tuples(
            [tuple(col) for col in expanded_columns.to_numpy()],
            names=expanded_columns.columns.tolist()
        )
        scores = pivoted.to_numpy()
        unique_counts = [len(expanded_columns[level].unique()) for level in expanded_columns.columns]
        scores = scores.reshape((pivoted.shape[0], *unique_counts))

    # Move instances dim to final dim
    scores = np.moveaxis(scores, 0, -1)

    if sorted:
        if len(col) == 1:
            sorted_indices = np.argsort(scores)
            columns = columns[sorted_indices]
            scores = scores[sorted_indices]
        else:
            mix_sums = scores.sum(axis=1)
            sorted_indices = mix_sums.argsort()[::-1]
            columns = columns[sorted_indices].tolist()
            scores = scores[sorted_indices]

    if not isinstance(columns, list):
        columns = columns.tolist()
    if not isinstance(index, list):
        index = index.tolist()

    if return_index:
        return index, columns, scores
    return columns, scores
