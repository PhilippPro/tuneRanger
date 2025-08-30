library(devtools)
setwd("./package")

load_all()
# roxygen2::roxygenise("../tuneRanger")
# install("../tuneRanger", dependencies = character(0))

#devtools::test()
devtools::check()
devtools::build()
#?submit_cran
devtools::release()
