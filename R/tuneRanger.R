#' tuneRanger
#' 
#' Automatic tuning of random forests of the \code{\link[ranger]{ranger}} package with one line of code. 
#'
#' @param task The mlr task created by \code{\link[mlr]{makeClassifTask}} or \code{\link[mlr]{makeRegrTask}}. 
#' @param measure Performance measure to evaluate. Default is auc for classification and mse for regression. Other possible performance measures can be looked up here: https://mlr-org.github.io/mlr-tutorial/release/html/performance/index.html
#' @param iters Number of iterations. Default is 70.
#' @param iters.warmup Number of iterations for the warmup. Default is 30. 
#' @param num.threads Number of threads. Default is number of CPUs available.
#' @param num.trees Number of trees.
#' @param parameters Optional list of fixed named parameters that should be passed to \code{\link[ranger]{ranger}}.
#' @param tune.parameters Optional character vector of parameters that should be tuned. 
#' Default is mtry, min.node.size and sample.fraction. Additionally replace and respect.unordered.factors can be 
#' included in the tuning process.
#' @param save.file.path File to which interim results are saved (e.g. "optpath.RData") in the current working directory. 
#' Default is NULL, which does not save the results. If a file was specified and one iteration fails the algorithm can be 
#' started again with \code{\link{restartTuneRanger}}.
#' @param build.final.model [\code{logical(1)}]\cr
#'   Should the best found model be fitted on the complete dataset?
#'   Default is \code{TRUE}. 
#' @import ranger mlr mlrMBO ParamHelpers BBmisc stats smoof lhs parallel
#' @importFrom DiceKriging km predict.km
#' @return list with recommended parameters and a data.frame with all evaluated hyperparameters and performance and time results for each run
#' @details Model based optimization is used as tuning strategy and the three parameters min.node.size, sample.fraction and mtry are tuned at once. Out-of-bag predictions are used for evaluation, which makes it much faster than other packages and tuning strategies that use for example 5-fold cross-validation. Classification as well as regression is supported. 
#' The measure that should be optimized can be chosen from the list of measures in mlr: http://mlr-org.github.io/mlr-tutorial/devel/html/measures/index.html
#' @seealso \code{\link{estimateTimeTuneRanger}} for time estimation and \code{\link{restartTuneRanger}} for continuing the algorithm if there was an error. 
#' @export
#' @examples 
#' \dontrun{
#' library(tuneRanger)
#' library(mlr)
#' 
#' # A mlr task has to be created in order to use the package
#' data(iris)
#' iris.task = makeClassifTask(data = iris, target = "Species")
#'  
#' # Estimate runtime
#' estimateTimeTuneRanger(iris.task)
#' # Tuning
#' res = tuneRanger(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
#'   num.threads = 2, iters = 70, save.file.path = NULL)
#'   
#' # Mean of best 5 % of the results
#' res
#' # Model with the new tuned hyperparameters
#' res$model
#' # Prediction
#' predict(res$model, newdata = iris[1:10,])}
tuneRanger = function(task, measure = NULL, iters = 70, iters.warmup = 30, num.threads = NULL, num.trees = 1000, 
  parameters = list(replace = FALSE, respect.unordered.factors = "order"), 
  tune.parameters = c("mtry", "min.node.size", "sample.fraction"), save.file.path = NULL,
  build.final.model = TRUE) {
  
  if(is.null(save.file.path)) {
    save.on.disk.at = NULL
  } else {
    save.on.disk.at = 1:(iters + 1)
  }
  
  fixed.param.in.tune = names(parameters) %in% tune.parameters
  if(any(fixed.param.in.tune))
    BBmisc::stopf("Fixed parameter %s cannot be tuning parameter at the same time.", names(parameters)[fixed.param.in.tune][1])
  
  type = getTaskType(task)
  size = getTaskSize(task)
  NFeats = getTaskNFeats(task)
  
  predict.type = ifelse(type == "classif", "prob", "response")
  if(is.null(measure)) {
    if(type == "classif") {
      cls.levels = getTaskClassLevels(task)
      if(length(cls.levels) == 2) {
        measure = list(brier)
      } else {
        measure = list(multiclass.brier)
      }
    }
    if(type == "regr") {
      measure = list(mse)
    }
  }
  measure.name = measure[[1]]$id
  minimize = measure[[1]]$minimize
  
  # Set the number of threads if not given by user
  if(is.null(num.threads))
    num.threads = parallel::detectCores()
  
  # Evaluation function
  performan = function(x) {
    par.vals = c(x, num.trees = num.trees, num.threads = num.threads, parameters)
    lrn = makeLearner(paste0(type, ".ranger"), par.vals = par.vals, predict.type = predict.type)
    mod = catchOrderWarning(mlr::train(lrn, task))
    preds = getOOBPreds(mod, task)
    performance(preds, measures = measure)
  }
  
  # Transformation of nodesize
  trafo_nodesize = function(x) ceiling((size * 0.2)^x)
  #trafo_mtry = function(x) round(NFeats^x)
  
  # Its ParamSet
  ps = makeParamSet(
    #makeNumericParam("mtry", lower = 0, upper = 1, trafo = trafo_mtry), 
    makeIntegerParam("mtry", lower = 1, upper = NFeats),
    makeNumericParam("min.node.size", lower = 0, upper = 1, trafo = trafo_nodesize), 
    makeNumericParam("sample.fraction", lower = 0.2, upper = 0.9),
    makeLogicalParam(id = "replace", default = FALSE),
    makeDiscreteLearnerParam("respect.unordered.factors", values = c("ignore", "order", "partition"), default = "order")
  )
  tunable.parameters = c("mtry", "min.node.size", "sample.fraction", "replace", "respect.unordered.factors")
  ps$pars = ps$pars[tunable.parameters %in% tune.parameters]
  
  # Budget
  f.evals = iters + iters.warmup
  mbo.init.design.size = iters.warmup
  
  # Focus search
  infill.opt = "focussearch"
  mbo.focussearch.points = iters + iters.warmup
  mbo.focussearch.maxit = 3
  mbo.focussearch.restarts = 3
  
  # The final SMOOF objective function
  objFun = smoof::makeMultiObjectiveFunction(
    name = "reg",
    fn = performan,
    par.set = ps,
    has.simple.signature = FALSE,
    noisy = TRUE,
    n.objectives = 1,
    minimize = minimize
  )
  
  # Build the control object
  method = "parego"
  if (method == "parego") {
    mbo.prop.points = 1
    mbo.crit = "cb"
    parego.crit.cb.pi = 0.5
  }
  
  control = makeMBOControl(n.objectives = 1L, propose.points = mbo.prop.points, # impute.y.fun = function(x, y, opt.path) 0.7, 
    save.on.disk.at = save.on.disk.at, save.file.path = save.file.path)
  control = setMBOControlTermination(control, max.evals = f.evals, iters = iters)
  control = setMBOControlInfill(control, #opt = infill.opt,
    opt.focussearch.maxit = mbo.focussearch.maxit,
    opt.focussearch.points = mbo.focussearch.points,
    opt.restarts = mbo.focussearch.restarts)
  
  design = generateDesign(mbo.init.design.size, getParamSet(objFun), fun = lhs::maximinLHS)
  #mbo.learner = makeLearner("regr.randomForest", predict.type = "se")
  mbo.learner = makeLearner("regr.km", covtype = "matern3_2", optim.method = "BFGS", nugget.estim = TRUE, 
    jitter = TRUE, predict.type = "se", config = list(show.learner.output = FALSE))
  
  result = mbo(fun = objFun, design = design, learner = mbo.learner, control = control)
  
  res = data.frame(result$opt.path)
  if("min.node.size" %in% tune.parameters)
    res$min.node.size = trafo_nodesize_end(res$min.node.size, size)
  #if("mtry" %in% tune.parameters)
  #  res$mtry = trafo_mtry_end(res$mtry, NFeats)
  colnames(res)[colnames(res) == "y"] = measure.name
  res = res[, c(tune.parameters, measure.name, "exec.time")]
  
  if (minimize) {
    recommended.pars = lapply(res[res[, measure.name] <= stats::quantile(res[, measure.name], 0.05),], summaryfunction)
  } else {
    recommended.pars = lapply(res[res[, measure.name] >= stats::quantile(res[, measure.name], 0.95),], summaryfunction)
  }
  recommended.pars = data.frame(recommended.pars)
  recommended.pars[colnames(res) %in% c("min.node.size", "mtry")] = round(recommended.pars[colnames(res) %in% c("min.node.size", "mtry")])
  
  # save the model with recommended hyperparameters
  mod = if(build.final.model) {
    ln.rec.pars = length(recommended.pars)
    x = as.list(recommended.pars[-c(ln.rec.pars - 1, ln.rec.pars)])
    x = c(x, num.trees = num.trees, num.threads = num.threads, parameters)
    lrn = mlr::makeLearner(paste0(type, ".ranger"), par.vals = x, predict.type = predict.type)
    catchOrderWarning(mlr::train(lrn, task))
  } else {
    NULL
  }
  
  out = list(recommended.pars = recommended.pars, results = res, model = mod)
  class(out) = "tuneRanger"
  return(out)
}

#' @export
print.tuneRanger = function(x, ...) {
  cat("Recommended parameter settings:", "\n")
  ln = length(x$recommended.pars)
  print(x$recommended.pars[-c(ln-1, ln)])
  cat("Results:", "\n")
  print(x$recommended.pars[c(ln-1, ln)])
}

trafo_mtry_end = function(x, NFeats) round(NFeats^x)
trafo_nodesize_end = function(x, size) ceiling((size * 0.2)^x)

summaryfunction = function(x) ifelse(class(x) %in% c("numeric", "integer"), mean(x), 
  names(sort(table(x), decreasing = TRUE)[1]))

catchOrderWarning = function(code) {
withCallingHandlers(code,
  warning = function(w) {
    if (grepl("Warning: The 'order' mode for unordered factor handling for multiclass classification is experimental.", w$message))
      invokeRestart("muffleWarning")
})
}
