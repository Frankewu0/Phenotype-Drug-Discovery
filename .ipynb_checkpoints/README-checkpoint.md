

## System Requirements
The source code developed in Python 3.8 using PyTorch 1.7.1. The required python dependencies are given below.

```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```
## Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name procat python=3.8
$ conda activate procat

# install requried python dependencies
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c dglteam dgl-cuda10.2==0.7.1
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install dgllife==0.2.8
$ pip install -U scikit-learn
$ pip install yacs
$ pip install prettytable


```

## Run PROCART on Our Data to Reproduce Results

To train PROCART, where we provide the basic configurations for all hyperparameters in `config.py`. For different in-domain and cross-domain tasks, the customized task configurations can be found in respective `configs/*.yaml` files.

For the in-domain experiments with vanilla PROCART, you can directly run the following command. `${dataset}` could either be `bindingdb`. `${split_task}` could be `random` and `cold`. 
```
$ python main.py --cfg "configs/PROCART.yaml" --data ${dataset} --split ${split_task}
```

For the cross-domain experiments with vanilla PROCART, you can directly run the following command. `${dataset}` could beither `bindingdb`.
```
$ python main.py --cfg "configs/PROCART.yaml" --data ${dataset} --split "cluster"
```
For the cross-domain experiments with PROCART, you can directly run the following command. `${dataset}` could beither `bindingdb`.
```
$ python main.py --cfg "configs/PROCART.yaml" --data ${dataset} --split "cluster"
```

