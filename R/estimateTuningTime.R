#' estimateTuneRFTime
#'
#' @param task
#' @param iters 
#' @param num.threads 
#' @param num.trees 
#' @param respect.unordered.factors 
#'
#' @return
#' @export
#'
#' @examples
estimateTuneRFTime = function(task, iters = 100, num.threads = 1, num.trees = 1000, respect.unordered.factors = TRUE) {
  type = getTaskType(task)
  predict.type = ifelse(type == "classif", "prob", "response")
  par.vals = list(num.trees = num.trees, num.threads = num.threads, respect.unordered.factors = TRUE, replace = TRUE)
  lrn = makeLearner(paste0(type, ".ranger"), par.vals = par.vals, predict.type = predict.type)
  time = system.time(train(lrn, task))[3]
  cat(paste("Approximated time for tuning:", my_seconds_to_period(time * iters + 100)))
}

#' estimateTuneRangerTime
#'
#' @param formula 
#' @param data 
#' @param iters 
#' @param num.threads 
#' @param num.trees 
#' @param respect.unordered.factors 
#'
#' @return
#' @export
#'
#' @examples
estimateTuneRangerTime = function(formula, data, iters = 100, num.threads = 1, num.trees = 1000, respect.unordered.factors = TRUE) {
  time = system.time(ranger(formula, data, num.threads = num.threads, 
    num.trees = num.trees, respect.unordered.factors = respect.unordered.factors))[3]
  cat(paste("Approximated time for tuning:", my_seconds_to_period(time * iters + 100)))
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