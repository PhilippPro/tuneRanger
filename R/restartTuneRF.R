#' restartTuneRF
#' 
#' Restarts the tuning process if an error occured. 
#'
#' @param save.file.path File to which interim results were saved. Default is optpath.RData in the current working 
#' @param task The mlr task created by \code{\link[mlr]{makeClassifTask}} or \code{\link[mlr]{makeRegrTask}}. 
#' @param measure Performance measure that was already used in the original \code{\link{tuneRF}} process. 
#' @return list with recommended parameters and data.frame with all evaluated hyperparameters and performance and time results for each run
#' @export
#' @examples 
#' library(tuneRF)
#' library(mlr)
#' 
#' # iris is a bit nonsense here
#' # A mlr task has to be created in order to use the package
#' # the already existing iris task is used here
#' unlink("./optpath.RData")
#' estimateTuneRFTime(iris.task)
#' res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
#'   num.threads = 8, iters = 100)
#' 
#' # Best 5 % of the results
#' results = res$results
#' results[results$multiclass.brier < quantile(results$multiclass.brier, 0.05),]
#' 
#' # Restart after failing in one of the iterations:
#' res = restartTuneRF("./optpath.RData", iris.task, measure = list(multiclass.brier))
#' 
restartTuneRF = function(save.file.path = "./optpath.RData", task, measure = NULL) {
  size = getTaskSize(task)
  res = mboContinue(save.file.path)
  if(!is.null(measure)) {
    measure.name = measure[[1]]$id
  } else {
    measure.name = "y"
  }
  res = data.frame(result$opt.path)
  res$min.node.size = trafo_nodesize_end(res$min.node.size, size)
  
  colnames(res)[colnames(res) == "y"] = measure.name
  res = res[, c("min.node.size", "sample.fraction", "mtry", measure.name, "exec.time")]
  recommended.pars = colMeans(res[res[, measure.name] < quantile(res[, measure.name], 0.05),])
  recommended.pars[c("min.node.size", "mtry")] = round(recommended.pars[c("min.node.size", "mtry")])
  
  list(recommended.pars = recommended.pars, results = results)
}