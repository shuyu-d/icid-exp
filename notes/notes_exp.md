## List of experiments todo 

Problem parameters: `d`, `deg`, `n`, `graph_type`, `sem_type` 

Algorithm parameters: `lambda_1` (for IC), `idec_lambda1` (for ID). 

### tests

Each item below represents one type of test to run (using ours and some
existing algorithms): 
    

* *Scalability* (incl. Golem): varying `d` in {100,..., 2000}, fix other problem parameters. 
  See `./exp_1_golem.py`. 

* *Sparsity of `B_true`* (incl. Golem): varying `deg` between 0.5 and 2, fix other problem parameters. 

* *Sensitivity to `n`* (optional): varying `n` between `d/2` (IC is not specifically prepared for this scenario) and `10d`.   

* *Selection of algorithm parameters* (only ours): varying (`lambda_1`, `idec_lambda1`) on a fixed set of problem parameters 

* *Graph types*: varying `graph_type` and `noise_type`, for at least one of the tests above. 

Each `.py` script will be named `./exp_<name>.py` (in the root folder); to be updated these 2~3 days.

#### Comparison with existing algorithms: 

* NOTEARS: available (included in `/external/`). Example: imported in `test_all.py`. 
* GOLEM: available on site Fujitsu 
* LiNGAM: to be added
* *some others to decide*: e.g. GES 

