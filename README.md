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

    # iris is a bit nonsense here
    unlink("./optpath.RData")
    estimateTuneRFTime(iris.task)
    res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
                 num.threads = 8, iters = 100)

    # Best 5 % of the results
    res[res$multiclass.brier < quantile(res$multiclass.brier, 0.05),]
#   min.node.size sample.fraction mtry multiclass.brier exec.time
#58             2       0.2468076    4       0.05905890     0.154
#82             4       0.2238066    3       0.05846804     0.168
#87             4       0.2449222    3       0.05906005     0.201
#91             2       0.2359518    3       0.05824274     0.216
#96             3       0.2481128    3       0.05892657     0.162
    ```
