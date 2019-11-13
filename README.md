# HGP-SL
Hierarchical Graph Pooling with Structure Learning (AAAI-2020).

This is a PyTorch implementation of the HGP-SL algorithm, which learns a low-dimensional representation for the entire graph. Specifically, the graph pooling operation utilizes node features and graph structure information to perform down-sampling on graphs. Then, a structure learning layer is stacked on the pooling operation, which aims to learn a refined graph structure that can best preserve the essential topological information.


## Requirements
* python3
* torch-scatter
* torch-sparse
* torch-cluster
* torch-geometric

Note:
This code repository is heavily built on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which is a Geometric Deep Learning Extension Library for PyTorch. Please refer [here](https://pytorch-geometric.readthedocs.io/en/latest/) for how to install and utilize the libary.


### Run
To run HGP-SL, just execute the following command for graph classification task:
```
python main.py
```

## Citing
If you find HGP-SL useful for your research, please consider citing the following paper:
```
@inproceedings{zhang2020hierarch,
  title={Hierarchical Graph Pooling with Structure Learning.},
  author={Zhen Zhang, Jiajun Bu, Martin Ester, Jianfeng Zhang, Chengwei Yao, Zhi Yu, Can Wang},
  booktitle={Proceedings of the 34th AAAI Conference on Artificial Intelligence},
  year={2020},
  organization={AAAI}
}
``` 
