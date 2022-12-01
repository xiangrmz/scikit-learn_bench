## Usage

### Dataset generation

run
```bash
python runner_datasets.py --configs configs/knn_clsf.json 
```
to generate datasets specified in config files in the config json.

### kdtree.py

This only run the program fit once without prediction. We use it for debugging.

The scripts may take **one** argument that specify how many cores is used for low-level parallelization. 
Use
```python
kdtree.py 112
```
to call the kd-tree knn with 112 cores.