#' restartTuneRanger
#' 
#' Restarts the tuning process if an error occured. 
#'
#' @param save.file.path File name in the current working directory to which interim results were saved by \code{\link{tuneRanger}}.
#' @param task The mlr task created by \code{\link[mlr]{makeClassifTask}} or \code{\link[mlr]{makeRegrTask}}. 
#' @param measure Performance measure that was already used in the original \code{\link{tuneRanger}} process. 
#' @return list with recommended parameters and data.frame with all evaluated hyperparameters and performance and time results for each run. 
#' No model is build.
#' @export
#' @examples 
#' \dontrun{
#' library(tuneRanger)
#' library(mlr)
#' 
#' # iris is a bit nonsense here
#' # A mlr task has to be created in order to use the package
#' # the already existing iris task is used here
#' estimateTimeTuneRanger(iris.task)
#' # temporarily file name to save results
#' path = tempfile()
#' res = tuneRanger(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
#'   num.threads = 8, iters = 70, save.file.path = path)
#' 
#' # Mean of best 5 % of the results
#' res
#' 
#' # Restart after failing in one of the iterations:
#' res = restartTuneRanger(save.file.path = path, iris.task, 
#' measure = list(multiclass.brier))}
restartTuneRanger = function(save.file.path = "optpath.RData", task, measure = NULL) {
  
  size = getTaskSize(task)
  res = mboContinue(save.file.path)
  if(!is.null(measure)) {
    measure.name = measure[[1]]$id
  } else {
    measure.name = "y"
  }
  
  res = data.frame(res$opt.path)
  if("min.node.size" %in% colnames(res))
    res$min.node.size = trafo_nodesize_end(res$min.node.size, size)
  colnames(res)[colnames(res) == "y"] = measure.name
  pos.measure.name = which(colnames(res) == measure.name)
  pos.exec.time = which(colnames(res) == "exec.time")
  res = res[, c(1:pos.measure.name, pos.exec.time)]
  
  recommended.pars = lapply(res[res[, measure.name] < stats::quantile(res[, measure.name], 0.05),], summaryfunction)
  recommended.pars = data.frame(recommended.pars)
  recommended.pars[colnames(res) %in% c("min.node.size", "mtry")] = round(recommended.pars[colnames(res) %in% c("min.node.size", "mtry")])
  
  out = list(recommended.pars = recommended.pars, results = res)
  class(out) = "tuneRanger"
  return(out)
}
