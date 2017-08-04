
# tuneRF: A package for tuning random forests

Philipp Probst

## Description
tuneRF is a package for automatic tuning of random forests with one line of code and hence intended for users that are not very familiar with tuning strategies. 

Model based optimization is used as tuning strategy and the three parameters min.node.size, sample.fraction and mtry are tuned at once. Out of bag predictions are used for evaluation, which makes it much faster than other packages and tuning strategies that use for example 5-fold cross-validation. Classification as well as regression is supported. 

The measure that should be optimized can be chosen from the list of measures in mlr: (http://mlr-org.github.io/mlr-tutorial/devel/html/measures/index.html)

The package is mainly based on [ranger](https://github.com/imbs-hl/ranger), [mlrMBO](http://mlr-org.github.io/mlrMBO/) and [mlr](https://github.com/mlr-org/mlr/#-machine-learning-in-r). 

## Installation
* Install the development version
    ```r
    devtools::install_github("PhilippPro/tuneRF")
    ```
    
## Usage
Quickstart:

    ```r
    library(tuneRF)
    library(mlr)

    # iris is a bit nonsense here
    # A mlr task has to be created in order to use the package
    # the already existing iris task is used here
    unlink("./optpath.RData")
    estimateTuneRFTime(iris.task)
    res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
                 num.threads = 8, iters = 100)

    # Best 5 % of the results
    res[res$multiclass.brier < quantile(res$multiclass.brier, 0.05),]
    ```
