# HGP-SL
Hierarchical Graph Pooling with Structure Learning (Preprint version is available on [arXiv](https://arxiv.org/abs/1911.05954)).

![](https://github.com/cszhangzhen/HGP-SL/blob/master/fig/model.png)

This is a PyTorch implementation of the HGP-SL algorithm, which learns a low-dimensional representation for the entire graph. Specifically, the graph pooling operation utilizes node features and graph structure information to perform down-sampling on graphs. Then, a structure learning layer is stacked on the pooling operation, which aims to learn a refined graph structure that can best preserve the essential topological information.


## Requirements
* python3.6
* pytorch==1.3.0
* torch-scatter==1.4.0
* torch-sparse==0.4.3
* torch-cluster==1.4.5
* torch-geometric==1.3.2

Note:
An older version of torch-sparse is needed, lower than 0.4.4. This code repository is heavily built on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which is a Geometric Deep Learning Extension Library for PyTorch. Please refer [here](https://pytorch-geometric.readthedocs.io/en/latest/) for how to install and utilize the library.

### Datasets
Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

This folder contains the following comma separated text files (replace DS by the name of the dataset):

**n = total number of nodes**

**m = total number of edges**

**N = number of graphs**

**(1) DS_A.txt (m lines)** 

*sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id)*

**(2) DS_graph_indicator.txt (n lines)**

*column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i*

**(3) DS_graph_labels.txt (N lines)** 

*class labels for all graphs in the dataset, the value in the i-th line is the class label of the graph with graph_id i*

**(4) DS_node_labels.txt (n lines)**

*column vector of node labels, the value in the i-th line corresponds to the node with node_id i*

There are OPTIONAL files if the respective information is available:

**(5) DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)**

*labels for the edges in DS_A_sparse.txt* 

**(6) DS_edge_attributes.txt (m lines; same size as DS_A.txt)**

*attributes for the edges in DS_A.txt* 

**(7) DS_node_attributes.txt (n lines)** 

*matrix of node attributes, the comma seperated values in the i-th line is the attribute vector of the node with node_id i*

**(8) DS_graph_attributes.txt (N lines)** 

*regression values for all graphs in the dataset, the value in the i-th line is the attribute of the graph with graph_id i*


### Run
To run HGP-SL, just execute the following command for graph classification task:
```
python main.py
```

### Parameter Settings
| Datasets      | lr        | weight_decay   | batch_size      | pool_ratio     | dropout  | net_layers |
| ------------- | --------- | -------------- | -------- 	   | --------       | -------- | ---------- |
| PROTEINS      | 0.001     | 0.001     	 | 512             | 0.5            | 0.0      | 3			| 
| Mutagenicity  | 0.001     | 0.001          | 512             | 0.8            | 0.0      | 3			|
| NCI109	    | 0.001     | 0.001          | 512             | 0.8            | 0.0      | 3			|
| NCI1          | 0.001		| 0.001          | 512             | 0.8            | 0.0      | 3			|
| DD            | 0.0001    | 0.001          | 64              | 0.3            | 0.5      | 2          |
| ENZYMES       | 0.001     | 0.001          | 128             | 0.8            | 0.0      | 2          |


## Citing
If you find HGP-SL useful for your research, please consider citing the following paper:
```
@article{zhang2019hierarchical,
  title={Hierarchical Graph Pooling with Structure Learning},
  author={Zhang, Zhen and Bu, Jiajun and Ester, Martin and Zhang, Jianfeng and Yao, Chengwei and Yu, Zhi and Wang, Can},
  journal={arXiv preprint arXiv:1911.05954},
  year={2019}
}
``` 
