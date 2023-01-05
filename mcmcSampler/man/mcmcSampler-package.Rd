\name{mcmcSampler-package}
\alias{mcmcSampler-package}
\alias{mcmcSampler}
\docType{package}
\title{
\packageTitle{mcmcSampler}
}
\description{
\packageDescription{mcmcSampler}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{mcmcSampler}
\packageIndices{mcmcSampler}
rwMCMC(target, init_theta, covmat, n_iterations)

adaptMCMC(
  target,
  init_theta,
  covmat,
  n_iterations,
  adapt_size_start,
  acceptance_rate_weight,
  acceptance_window,
  info,
  seedMH = 1,
  eps = 0.05

rtnorm_gibbs(n, mu, sigma, a, b)
)
}
\author{
\packageAuthor{mcmcSampler}

Maintainer: \packageMaintainer{mcmcSampler}
}

\keyword{ package }

\examples{
library(mcmcSampler)
p.log <- function(x) {
  B <- 0.03 # controls 'bananacity'
  -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2
}
sample <- rwMCMC(target=p.log, init_theta=c(10,10), covmat=diag(c(1,1)),n_iterations=1e4)

rtnorm_gibbs(n=100,mu=0, sigma=1, a=2, b=4)

banana1 <- adaptMCMC(target=p.log,init_theta=c(10,10),covmat=diag(c(1,1)),n_iterations=1e3,adapt_size_start=10,acceptance_rate_weight=0,acceptance_window=0,info=1e2, seedMH = 123)
}
}
