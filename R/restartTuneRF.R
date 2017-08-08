#' restartTuneRF
#'
#' @param save.file.path 
#' @param task 
#' @param measure 
#'
#' @return
#' @export
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
  recommendation = colMeans(res[res[, measure.name] < quantile(res[, measure.name], 0.05),])
  recommendation[c("min.node.size", "mtry")] = round(recommendation[c("min.node.size", "mtry")])
  
  list(recommendation = recommendation, results = results)
}