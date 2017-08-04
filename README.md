# tuneRF
Automatic tuning of random forests with model based optimization

* Install the development version
    ```r
    devtools::install_github("PhilippPro/tuneRF")
    ```
* Quickstart:

    ```r
    library(tuneRF)
    library(mlr)
    res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
                 num.threads = 8, iters = 100)
    # Best 5 % of the results
    res[res$y < quantile(res$y, 0.05),]
    ```
