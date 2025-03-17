import numpy as np
import pandas as pd
import warnings
import duckdb

def load_db_backend(local_path):
    con = duckdb.connect()

    # WEKA optimizations
    con.execute("PRAGMA threads=200")
    con.execute("PRAGMA enable_object_cache=true")
    # con.execute("PRAGMA enable_profiling='json'")
    con.execute("PRAGMA memory_limit='1.5TB'")
    con.execute("PRAGMA threads=200").fetchall()

    # Use PARALLEL to enable multi-threaded reading
    con.execute(f"""
    CREATE TABLE instances AS 
    SELECT * FROM read_parquet('{local_path}')
    """)
    # ORDER BY task, model, step, mix

    # Create index for commonly used query
    con.execute(f"""
    CREATE INDEX idx_task_model ON instances (task, model);
    """)
    return con


def get_slice_db(db, mix=None, model=None, task=None, step=None):
    """ Index to return a df of some (data mix, model, task, step) """
    # Build SQL query conditions
    conditions = []
    if task:
        tasks = [task] if isinstance(task, str) else task
        if len(tasks) == 1:
            conditions.append(f"task = '{tasks[0]}'")
        else:
            conditions.append("task IN " + str(tuple(tasks)))
    if model:
        models = [model] if isinstance(model, str) else model
        if len(models) == 1:
            conditions.append(f"model = '{models[0]}'")
        else:
            conditions.append("model IN " + str(tuple(models)))
    if step is not None:
        steps = [step] if isinstance(step, int) else step
        if len(steps) == 1:
            conditions.append(f"step = {steps[0]}")
        else:
            conditions.append("step IN " + str(tuple(steps)))
    if mix:
        mixes = [mix] if isinstance(mix, str) else mix
        if len(mixes) == 1:
            conditions.append(f"mix = '{mixes[0]}'")
        else:
            conditions.append("mix IN " + str(tuple(mixes)))

    # Build and execute query
    # query = "SELECT * FROM instances"
    query = "SELECT task, model, step, mix, primary_score, logits_per_byte_corr, native_id from instances"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    # if 'step' in db.execute("SELECT * FROM instances LIMIT 1").df().columns:
    #     query += " ORDER BY step"

    # df = db.execute(query).df() # convert result to df
    df = db.execute(query)
    df = df.arrow().to_pandas() # convert result to df
    
    # Re-arrange first cols
    first_cols = ['task', 'model', 'step', 'mix', 'primary_score', 'logits_per_byte_corr'] 
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]

    return df

def get_instance_db(db, instance_id):
    """ Index to return a df of some (instance_id) """
    instance_ids = [instance_id] if isinstance(instance_id, str) else instance_id
    
    query = "SELECT * FROM instances WHERE instance_id IN "
    query += str(tuple(instance_ids)) if len(instance_ids) > 1 else f"('{instance_ids[0]}')"
    
    if 'step' in db.execute("SELECT * FROM instances LIMIT 1").df().columns:
        query += " ORDER BY step"

    df = db.execute(query).df()
    return df

def get_max_k_step_db(_slice, k=1):
    """Filter for only rows with the top k steps."""
    top_steps = _slice['step'].nlargest(k).unique()
    step_filter = _slice['step'].isin(top_steps)
    _slice = _slice[step_filter]
    return _slice


# def get_nd_array_db(db, col, metric, mix=None, model=None, task=None, step=None, sorted=False, return_index=False):
#     """ Get an nd array of (COL, instances), sorted by overall performance """
#     col = [col] if not isinstance(col, list) else col
    
#     use_max_step = False
#     if step == 'max':
#         use_max_step = True
#         step = None
    
#     # Build SQL query conditions
#     conditions = []
#     if task:
#         tasks = [task] if isinstance(task, str) else task
#         if len(tasks) == 1:
#             conditions.append(f"task = '{tasks[0]}'")
#         else:
#             conditions.append("task IN " + str(tuple(tasks)))
#     if model:
#         models = [model] if isinstance(model, str) else model
#         if len(models) == 1:
#             conditions.append(f"model = '{models[0]}'")
#         else:
#             conditions.append("model IN " + str(tuple(models)))
#     if step is not None:
#         steps = [step] if isinstance(step, int) else step
#         if len(steps) == 1:
#             conditions.append(f"step = {steps[0]}")
#         else:
#             conditions.append("step IN " + str(tuple(steps)))
#     if mix:
#         mixes = [mix] if isinstance(mix, str) else mix
#         if len(mixes) == 1:
#             conditions.append(f"mix = '{mixes[0]}'")
#         else:
#             conditions.append("mix IN " + str(tuple(mixes)))

#     # Base query to get unique combinations and values
#     base_query = f"""
#     SELECT {', '.join(col)},
#            native_id || ':' || task as unique_id,
#            {metric}
#     FROM instances
#     """
#     if conditions:
#         base_query += " WHERE " + " AND ".join(conditions)

#     # Get unique values for each column to build the structure
#     col_values = {}
#     for c in col:
#         values_query = f"SELECT DISTINCT {c} FROM ({base_query}) ORDER BY {c}"
#         col_values[c] = [r[0] for r in db.execute(values_query).fetchall()]

#     if not col_values[col[0]]:  # No data found
#         if return_index:
#             return [], [], np.array([])
#         return [], np.array([])

#     # Get all unique IDs in order
#     id_query = f"SELECT DISTINCT unique_id FROM ({base_query}) ORDER BY unique_id"
#     unique_ids = [r[0] for r in db.execute(id_query).fetchall()]

#     # For single column case, use simpler query
#     if len(col) == 1:
#         query = f"""
#         SELECT {col[0]},
#                array_agg({metric} ORDER BY unique_id) as values
#         FROM ({base_query})
#         GROUP BY {col[0]}
#         ORDER BY {col[0]}
#         """
#         result = db.execute(query).fetchall()
#         columns = [r[0] for r in result]
#         scores = np.array([list(r[1]) for r in result])

#     else:
#         # For multiple columns, build a query that creates the full cartesian product
#         cross_query_parts = []
#         for c in col:
#             cross_query_parts.append(f"SELECT unnest(array{col_values[c]}) as {c}")
#         cross_query = " CROSS JOIN ".join(f"({part})" for part in cross_query_parts)

#         # Join with the data and aggregate
#         query = f"""
#         WITH CartesianProduct AS ({cross_query}),
#              Data AS ({base_query})
#         SELECT {', '.join(f'cp.{c}' for c in col)},
#                array_agg(COALESCE(Data.{metric}, NULL) ORDER BY Data.unique_id) as values
#         FROM CartesianProduct cp
#         LEFT JOIN Data ON {' AND '.join(f'cp.{c} = Data.{c}' for c in col)}
#         GROUP BY {', '.join(f'cp.{c}' for c in col)}
#         ORDER BY {', '.join(f'cp.{c}' for c in col)}
#         """
        
#         result = db.execute(query).fetchall()
#         columns = [tuple(r[:-1]) for r in result]
#         scores = np.array([list(r[-1]) for r in result])

#     # Reshape for multiple columns
#     if len(col) > 1:
#         shape = [len(col_values[c]) for c in col] + [len(unique_ids)]
#         scores = scores.reshape(shape)
#         scores = np.moveaxis(scores, -1, -1)  # Keep instances in last dimension

#     if sorted and len(col) > 1:
#         mix_sums = scores.sum(axis=1)
#         sorted_indices = mix_sums.argsort()[::-1]
#         columns = [columns[i] for i in sorted_indices]
#         scores = scores[sorted_indices]

#     if return_index:
#         return unique_ids, columns, scores
#     return columns, scores


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
