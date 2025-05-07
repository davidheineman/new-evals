import os, time, sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# # last 30 ckpts of 1.5B-4T OLMo 2 run
# GCS_STEPS = [1610000, 1620000, 1630000, 1640000, 1650000, 1660000, 1670000, 1680000, 1690000, 1700000, 1710000, 1720000, 1730000, 1740000, 1750000, 1760000, 1770000, 1780000, 1790000, 1800000, 1810000, 1820000, 1830000, 1840000, 1850000, 1860000, 1870000, 1880000, 1890000, 1900000, 1907359]
# GCS_CHECKPOINTS = [f'gs://ai2-llm/checkpoints/OLMo-small/peteish1/step{step}' for step in GCS_STEPS]

# # Varying data seed
# GCS_NAMES = [
#     'OLMo2-data-seed-01345-1B-5xC',
#     'OLMo2-data-seed-10294-1B-5xC',
#     'OLMo2-data-seed-23095-1B-5xC',
#     'OLMo2-data-seed-39240-1B-5xC',
#     'OLMo2-data-seed-59430-1B-5xC',
#     'OLMo2-data-seed-60439-1B-5xC',
#     'OLMo2-data-seed-89632-1B-5xC'
# ]

# Backup models
GCS_NAMES = [
    'peteish-moreeval-3B-0.5xC/',
    'peteish-moreeval-3B-10xC/',
    'peteish-moreeval-3B-1xC/',
    'peteish-moreeval-3B-2xC/',
    'peteish-moreeval-3B-5xC/',
]
GCS_CHECKPOINTS = [f'gs://ai2-llm/checkpoints/davidh/OLMo-ladder/{name}' for name in GCS_NAMES]

# Create rsync commands
from concurrent.futures import ThreadPoolExecutor

def check_and_download(gcs_path):
    local_dir = f'/oe-eval-default/{gcs_path[len("gs://"):].replace("/ladder/checkpoints/", "/seed/")}'
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
        
    command = f"""\
        gsutil -m rsync -r {gcs_path} {local_dir}/ \
    """
    print(f'Executing command:\n{command}')
    os.system(command)

with ThreadPoolExecutor() as executor:
    executor.map(check_and_download, GCS_CHECKPOINTS)
