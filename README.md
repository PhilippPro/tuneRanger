
# tuneRanger: A package for tuning random forests

Philipp Probst

## Description
**tuneRanger** is a package for automatic tuning of random forests with one line of code and intended for users that want to get the best out of their random forest model. 

Model based optimization is used as tuning strategy and the three parameters min.node.size, sample.fraction and mtry are tuned at once. Out-of-bag predictions are used for evaluation, which makes it much faster than other packages and tuning strategies that use for example 5-fold cross-validation. Classification as well as regression is supported. 

The measure that should be optimized can be chosen from the list of measures in mlr: https://mlr-org.github.io/mlr/articles/measures.html

The package is mainly based on [ranger](https://github.com/imbs-hl/ranger), [mlrMBO](http://mlr-org.github.io/mlrMBO/) and [mlr](https://github.com/mlr-org/mlr/#-machine-learning-in-r). 

The package is also described in an arXiv-Paper: [https://arxiv.org/abs/1804.03515](https://arxiv.org/abs/1804.03515)


## Installation
The development version

    devtools::install_github("mlr-org/mlr")
    devtools::install_github("PhilippPro/tuneRanger")
    
 CRAN

    install.packages("tuneRanger")
    
## Usage
Quickstart:

    library(tuneRanger)
    library(mlr)

    # A mlr task has to be created in order to use the package
    # We make an mlr task with the iris dataset here 
    # (Classification task with makeClassifTask, Regression Task with makeRegrTask)
    iris.task = makeClassifTask(data = iris, target = "Species")
    
    # Rough Estimation of the Tuning time
    estimateTimeTuneRanger(iris.task)

    # Tuning process (takes around 1 minute); Tuning measure is the multiclass brier score
    res = tuneRanger(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
                 num.threads = 2, iters = 70)
 
    # Mean of best 5 % of the results
    res
    # Model with the new tuned hyperparameters
    res$model

    # Restart after failing in one of the iterations:
    res = restartTuneRanger("./optpath.RData", iris.task, measure = list(multiclass.brier))

## Benchmark

You can see a benchmark in the [paper](https://arxiv.org/abs/1804.03515). 

Moreover, for regression I compared three different tuning implementations (**tuneRanger**, (autoxgboost)[https://github.com/ja-thomas/autoxgboost] and (liquidSVM)[https://github.com/liquidSVM/liquidSVM] on 29 regression tasks. 
in their default mode and the default of **ranger**. The results show the competitiveness of **tuneRanger** and can be seen in the following graphs:

![R-Squared](https://github.com/PhilippPro/tuneRanger/blob/master/benchmark/figure/rsq_results.pdf)
![Spearmans-Rho](https://github.com/PhilippPro/tuneRanger/blob/master/benchmark/figure/spearman_results.pdf)
![Training time](https://github.com/PhilippPro/tuneRanger/blob/master/benchmark/figure/time_results.pdf)

<img align="center" src="hhttps://github.com/PhilippPro/tuneRanger/blob/master/benchmark/figure/rsq_results.pdf" width="1000px">

A disadvantage is the long runtime (e.g. compared to liquidSVM), improvements could be made on this issue.

## How to cite

Please cite the paper, if you use the package:

```bibtex
@ARTICLE{tuneRanger,
  author = {Probst, Philipp and Wright, Marvin and Boulesteix, Anne-Laure}, 
  title = {Hyperparameters and Tuning Strategies for Random Forest},
  journal = {ArXiv preprint arXiv:1804.03515},
  archivePrefix = "arXiv",
  eprint = {1804.03515},
  primaryClass = "stat.ML",
  keywords = {Statistics - Machine Learning, Computer Science - Learning},
  year = 2018,
  url = {https://arxiv.org/abs/1804.03515}
}
```
