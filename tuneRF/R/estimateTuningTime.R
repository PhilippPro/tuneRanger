
estimateTuneRangerTime = function(formula, data, iters = 100, num.threads = 1, num.trees = 1000, respect.unordered.factors = TRUE) {
  time = system.time(ranger(formula, data, num.threads = num.threads, 
    num.trees = num.trees, respect.unordered.factors = respect.unordered.factors))[3]
  cat(paste("Time estimation is", my_seconds_to_period(time * iters)))
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