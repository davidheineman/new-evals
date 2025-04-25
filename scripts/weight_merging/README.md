```sh
pip install -r requirements.txt

# Merge models
python merge.py --model-list-file last-30/32B.txt
python merge.py --model-list-file last-30/13B.txt
python merge.py --model-list-file last-30/7B.txt

python merge.py --model-list-file last-5/32B.txt
python merge.py --model-list-file last-5/13B.txt
python merge.py --model-list-file last-5/7B.txt
```