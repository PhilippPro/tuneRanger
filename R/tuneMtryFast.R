#' tuneMtryFast
#' 
#' Similar to tuneRF in \code{\link[randomForest]{randomForest}} but for \code{\link[ranger]{ranger}}.
#' 
#' Provides fast tuning for the mtry hyperparameter. 
#' 
#' Starting with the default value of mtry, search for the optimal value (with respect to Out-of-Bag error estimate) of mtry for randomForest.
#'
#' @param formula Object of class formula or character describing the model to fit. Interaction terms supported only for numerical variables.
#' @param data Training data of class data.frame, matrix, dgCMatrix (Matrix) or gwaa.data (GenABEL).
#' @param mtryStart starting value of mtry; default is the same as in \code{\link[ranger]{ranger}}
#' @param num.treesTry number of trees used at the tuning step
#' @param stepFactor at each iteration, mtry is inflated (or deflated) by this value
#' @param improve the (relative) improvement in OOB error must be by this much for the search to continue
#' @param trace whether to print the progress of the search
#' @param plot whether to plot the OOB error as function of mtry
#' @param doBest whether to run a forest using the optimal mtry found
#' @param ... options to be given to \code{\link[ranger]{ranger}}
#' @importFrom graphics axis
#' 
#' @return If doBest=FALSE (default), it returns a matrix whose first column contains the mtry values searched, and the second column the corresponding OOB error.
#' 
#' If doBest=TRUE, it returns the \code{\link[ranger]{ranger}} object produced with the optimal mtry.
#' @export
#'
#' @examples
#' library(tuneRanger)
#' 
#' data(iris)
#' res <- tuneMtryFast(Species ~ ., data = iris, stepFactor = 1.5)
tuneMtryFast = function (formula, data, mtryStart = floor(sqrt(ncol(data)-1)), 
  num.treesTry = 50, stepFactor = 2, improve = 0.05, trace = TRUE, plot = TRUE, doBest = FALSE, ...) 
{
  if (improve < 0) 
    stop("improve must be non-negative.")
  
  #classRF = is.factor(y)
  nvar = ncol(data)-1
  
  errorOld = ranger(formula, data, mtry = mtryStart, num.trees = num.treesTry, ...)$prediction.error
  
  if (trace) {
    cat("mtry =", mtryStart, " OOB error =", errorOld, "\n")
  }
  
  oobError = list()
  oobError[[1]] = errorOld
  names(oobError)[1] = mtryStart
  for (direction in c("left", "right")) {
    if (trace)
      cat("Searching", direction, "...\n")
    Improve = 1.1 * improve
    mtryBest = mtryStart
    mtryCur = mtryStart
    while (Improve >= improve) {
      mtryOld = mtryCur
      mtryCur = if (direction == "left") {
        max(1, ceiling(mtryCur/stepFactor))
      }
      else {
        min(nvar, floor(mtryCur * stepFactor))
      }
      if (mtryCur == mtryOld) 
        break
      errorCur = ranger(formula, data, mtry = mtryStart, num.trees = num.treesTry, ...)$prediction.error
      if (trace) {
        cat("mtry =", mtryCur, "\tOOB error =", errorCur, "\n")
      }
      oobError[[as.character(mtryCur)]] = errorCur
      Improve = 1 - errorCur/errorOld
      cat(Improve, improve, "\n")
      if (Improve > improve) {
        errorOld = errorCur
        mtryBest = mtryCur
      }
    }
  }
  mtry = sort(as.numeric(names(oobError)))
  res = unlist(oobError[as.character(mtry)])
  res = cbind(mtry = mtry, OOBError = res)
  if (plot) {
    plot(res, xlab = expression(m[try]), ylab = "OOB Error", 
      type = "o", log = "x", xaxt = "n")
    axis(1, at = res[, "mtry"])
  }
  if (doBest) 
    res = ranger(formula, data, mtry = res[which.min(res[,2]), 1], ...)
  return(res)
}
