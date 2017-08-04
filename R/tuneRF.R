#' tuneRF
#'
#' @param task The mlr task created by \code{\link[mlr]{makeClassifTask}} or \code{\link[mlr]{makeRegrTask}}. 
#' @param measure Performance measure to evaluate. Default is auc for classification and mse for regression. Other possible performance measures can be looked up here: https://mlr-org.github.io/mlr-tutorial/release/html/performance/index.html
#' @param iters Number of iterations. 
#' @param num.threads Number of threads. Default is 1.
#' @param num.trees Number of trees.
#' @param replace Sample with replacement.
#' @param save.file.path File to which interim results are saved.
#' @return data.frame with all evaluated hyperparameters and performance and time results for each run
#' @export
#' @examples 
#' library(tuneRF)
#' library(mlr)
#' # iris is a bit nonsense here
#' # A mlr task has to be created in order to use the package
#' 
#' # the already existing iris task is used here
#' unlink("./optpath.RData")
#' estimateTuneRFTime(iris.task)
#' 
#' res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, 
#'   num.threads = 8, iters = 100)
#'   
#' # Best 5 % of the results
#' res[res$multiclass.brier < quantile(res$multiclass.brier, 0.05),]
tuneRF = function(task, measure = NULL, iters = 100, num.threads = 1, num.trees = 1000, replace = TRUE, 
  save.file.path = "./optpath.RData") {
  
  type = getTaskType(task)
  size = getTaskSize(task)
  NFeats = getTaskNFeats(task)
  predict.type = ifelse(type == "classif", "prob", "response")
  if(is.null(measure)) {
    if(type == "classif") {
      measure = list(auc)
    }
    if(type == "regr") {
      measure = list(mse)
    }
  }
  minimize = measure[[1]]$minimize
  
  # Evaluation function
  performan = function(x) {
    x = c(x, num.trees = num.trees, num.threads = num.threads, respect.unordered.factors = TRUE, replace = TRUE)
    lrn = makeLearner(paste0(type, ".ranger"), par.vals = x, predict.type = predict.type)
    
    mod = train(lrn, task)
    preds = getOOBPreds(mod, task)
    performance(preds, measures = measure)
  }
  
  trafo_nodesize = function(x) ceiling(2^(log(size, 2) * x))
  
  # Its ParamSet
  ps = makeParamSet(
    makeNumericParam("min.node.size", lower = 0, upper = 1, trafo = trafo_nodesize), 
    makeNumericParam("sample.fraction", lower = 0.2, upper = 0.9),
    makeIntegerParam("mtry", lower = 1, upper = NFeats)
  )
  
  # Budget
  f.evals = iters
  mbo.init.design.size = 30
  
  # Focus search
  infill.opt = "focussearch"
  mbo.focussearch.points = iters
  mbo.focussearch.maxit = 3
  mbo.focussearch.restarts = 3
  
  library(mlrMBO)
  # The final SMOOF objective function
  objFun = makeMultiObjectiveFunction(
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
  
  control = makeMBOControl(n.objectives = 1L, propose.points = mbo.prop.points, impute.y.fun = function(x, y, opt.path) 0.7, 
    save.on.disk.at = 1:10, save.file.path = save.file.path)
  control = setMBOControlTermination(control, max.evals =  f.evals, iters = 300)
  control = setMBOControlInfill(control, #opt = infill.opt,
    opt.focussearch.maxit = mbo.focussearch.maxit,
    opt.focussearch.points = mbo.focussearch.points,
    opt.restarts = mbo.focussearch.restarts)
  
  #mbo.learner = makeLearner("regr.randomForest", predict.type = "se")
  
  design = generateDesign(mbo.init.design.size, getParamSet(objFun), fun = lhs::maximinLHS)
  
  set.seed(123)
  result = mbo(fun = objFun, design = design, learner = mbo.learner, control = control)
  
  res = data.frame(result$opt.path)
  res$min.node.size = trafo_nodesize(res$min.node.size)
  
  colnames(res)[4] = measure[[1]]$id
  res[, c("min.node.size", "sample.fraction", "mtry", measure[[1]]$id, "exec.time")]
}


