# High-dimensional causal discovery from Inverse Covariance matrices by Independence-based Decomposition (ICID).

This package implements the ICID algorithm, which consists of the
following two consecutive steps:

    (IC) S* = argmin loss(S) - logdet(S) + lambda_1 |S|_1,
                subject to S > 0 (symmetric positive definite),

    (ID) B* = argmin |S* - phi(B)|^2 + lambda_1' |B|_1,
                subject to B in DAG and supp(B) \subset supp(S).

For the ID problem, phi(B) = (1/s)(I-B)(I-B)' is a quadratic matrix function of B.


## Core functions  

- `icid/icid.py` - Implementation of the ICID algorithm 
- `icid/SparseMatDecomp.py` - Computation of the Independence-based Decomposition of an input inverse covariance matrix 
- `icid/Loram.py` - Computation of proximal mappings with respect to the DAG characteristic function h. 


## Requirements

- Python 3.6+
- `numpy`
- `scipy`
- `scikit-learn` : the module `GraphLasso` (named `GraphicalLasso` in more recent version of scikit-learn) is used for the step (IC).
- `python-igraph`: Install [igraph C core](https://igraph.org/c/) and `pkg-config` first.
- `pywt`: for the soft-threshod function (TODO: copy this function individually to remove this unnecessary dependency) 
- `NOTEARS/utils.py` - graph simulation, data simulation, and accuracy evaluation from [Zheng et al. 2018]


## Running a demo

```bash
$ cd icid-exp/
$ make 
$ python exp_2f.py icid # run ICID
$ python exp_2f.py ideal # run O-ICID
```

