library(mlr)
library(ranger)

context("Output check")

test_that("classification ranger", {
  library(tuneRF)
  library(mlr)

  # A mlr task has to be created in order to use the package
  # the already existing iris task is used here
  unlink("./optpath.RData")
  estimateTimeTuneRF(iris.task, num.trees = 100, num.threads = 1)

  res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 1, iters = 5)
  
  expect_true(is.data.frame(res$results))
  expect_true(is.data.frame(res$recommended.pars))
  expect_true(class(res$model) == "WrappedModel")
})
