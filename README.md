
# tuneRF: A package for tuning random forests

Philipp Probst

## Description
tuneRF is a package for automatic tuning of random forests with one line of code and intended for users that are not very familiar with tuning strategies. 

Model based optimization is used as tuning strategy and the three parameters min.node.size, sample.fraction and mtry are tuned at once. Out-of-bag predictions are used for evaluation, which makes it much faster than other packages and tuning strategies that use for example 5-fold cross-validation. Classification as well as regression is supported. 

The measure that should be optimized can be chosen from the list of measures in mlr: http://mlr-org.github.io/mlr-tutorial/devel/html/measures/index.html

The package is mainly based on [ranger](https://github.com/imbs-hl/ranger), [mlrMBO](http://mlr-org.github.io/mlrMBO/) and [mlr](https://github.com/mlr-org/mlr/#-machine-learning-in-r). 

## Installation
The development version

    
    devtools::install_github("PhilippPro/tuneRF")
    
    
## Usage
Quickstart:

    library(tuneRF)
    library(mlr)

    # A mlr task has to be created in order to use the package
    # the already existing iris task is used here
    estimateTuneRFTime(iris.task)
    res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
                 num.threads = 8, iters = 100)
    res

    # Best 5 % of the results
    results = res$results
    results[results$multiclass.brier >= quantile(results$multiclass.brier, 0.95),]


    # Restart after failing in one of the iterations:
    res = restartTuneRF("./optpath.RData", iris.task, measure = list(multiclass.brier))
