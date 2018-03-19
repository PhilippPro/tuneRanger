#' estimateTimeTuneRanger
#'
#' @param task The mlr task created by makeClassifTask or makeRegrTask. 
#' @param iters Number of iterations. 
#' @param num.threads Number of threads. Default is 1.
#' @param num.trees Number of trees.
#' @param respect.unordered.factors Handling of unordered factor covariates. One of 'ignore', 'order' and 'partition'. 'order' is the default.
#' @return estimated time for the tuning procedure
#' @importFrom methods slot<-
#' @importFrom lubridate period
#' @export
#' @examples
#' estimateTimeTuneRanger(iris.task)
estimateTimeTuneRanger = function(task, iters = 100, num.threads = 1, num.trees = 1000, respect.unordered.factors = "order") {
  type = getTaskType(task)
  NFeats = getTaskNFeats(task)
  mtry = ceiling(NFeats/2)
  predict.type = ifelse(type == "classif", "prob", "response")
  par.vals = list(num.trees = num.trees, num.threads = num.threads, respect.unordered.factors = respect.unordered.factors, replace = FALSE, mtry = mtry)
  lrn = makeLearner(paste0(type, ".ranger"), par.vals = par.vals, predict.type = predict.type)
    # Train model and avoid the nasty error message from ranger
  time =  system.time(mod <- catchOrderWarning(mlr::train(lrn, task)))[3]
  
  cat(paste("Approximated time for tuning:", my_seconds_to_period(time * iters + 50)))
  invisible(time*iters + 50)
}

my_seconds_to_period = function(x) {
  # from lubridate package
  span <- as.double(x)
  remainder <- abs(span)
  newper <- period(second = rep(0, length(x)))
  slot(newper, "day") <- remainder%/%(3600 * 24)
  remainder <- remainder%%(3600 * 24)
  slot(newper, "hour") <- remainder%/%(3600)
  remainder <- remainder%%(3600)
  slot(newper, "minute") <- remainder%/%(60)
  slot(newper, ".Data") <- round(remainder%%(60),0)
  newper * sign(span)
}


