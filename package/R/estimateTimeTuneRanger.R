#' estimateTimeTuneRanger
#'
#' @param task The mlr task created by makeClassifTask or makeRegrTask. 
#' @param iters Number of iterations. 
#' @param num.threads Number of threads. Default is 1.
#' @param num.trees Number of trees.
#' @param respect.unordered.factors Handling of unordered factor covariates. One of 'ignore', 'order' and 'partition'. 'order' is the default.
#' @return estimated time for the tuning procedure
#' @importFrom methods slot<-
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
  days = round(x %/% (60 * 60 * 24))
  hours = round((x - days*60*60*24) %/% (60 * 60))
  minutes = round((x - days*60*60*24 - hours*60*60) %/% 60)
  seconds = round(x - days*60*60*24 - hours*60*60 - minutes*60)
  days_str = ifelse(days == 0, "", paste0(days, "d "))
  hours_str = ifelse((hours == 0 & days == 0), "", paste0(hours, "H "))
  minutes_str = ifelse((minutes == 0 & days == 0 & hours == 0), "", paste0(minutes, "M "))
  seconds_str = paste0(seconds, "S")
  final_str = paste0(days_str, hours_str, minutes_str, seconds_str)
  return(final_str)
}
