# Co-Regularized Deep Multi-Network Embedding

This is a reference implementation for DMNE. The DMNE algorithm learns node representations on multi-network data. Please refer to the following paper for details.

Reference:
> [Co-Regularized Deep Multi-Network Embedding](http://www.personal.psu.edu/jzn47/paper/www18_dmne.pdf)<br>
> Jingchao Ni, Shiyu Chang, Xiao Liu, Wei Cheng, Haifeng Chen, Dongkuan Xu and Xiang Zhang<br>
> Proceedings of the International Conference on World Wide Web (WWW), 2018.

For any questions about the code, please contact Jingchao Ni (jingchaoni@psu.edu).

## Input

The format of the input data is the edge list of each network.

**Domain network**

	node_id node_id edge_weight

For undirected networks, the same edge will be written in two directions, e.g., 1, 2, 1.00 and 2, 1, 1.00.

**Cross-network relationship**

	node_id_in_domain_1 node_id_in_domain_2 relationship_weight

**Label in each domain (for evaluation)**

	node_id label

## Output

For each network, there is an output file in ``emb/``. If a network has n nodes, there are n+1 lines in its output file. The first line contains the number of nodes and the dimensionality of the embeddings.

	num_of_nodes dim_of_embedding

The next n lines contain node embeddings.

	node_id dim_1 dim_2 ... dim_d

where dim_1, ..., dim_d are the d-dimensional embedding of a node.

## Running

* Install libsvm in ``libsvm/``.
* Run ``rundemo.m`` to see the demo program on 6ng dataset.

## Citing

If you find DMNE useful for your research, please consider citing the following paper:

	@inproceedings{ni2018co,
	  title={Co-Regularized Deep Multi-Network Embedding},
	  author={Ni, Jingchao and Chang, Shiyu and Liu, Xiao and Cheng, Wei and Chen, Haifeng and Xu, Dongkuan and Zhang, Xiang},
	  booktitle={Proceedings of the International Conference on World Wide Web (WWW)},
	  pages={469--478},
	  year={2018},
	  organization={International World Wide Web Conferences Steering Committee}
	}
