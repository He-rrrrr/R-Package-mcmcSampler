#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <random>

using namespace Rcpp;

double mvrnorm_pdff(arma::vec x, arma::vec mu, arma::mat sigma) {

  //define constants
  int k = x.size();
  double twoPi = 2 * M_PI;
  double out;

  double rootTerm;
  rootTerm = 1 / sqrt(pow(twoPi, k) * det(sigma));
  arma::mat AexpTerm;
  AexpTerm = exp(-0.5 * arma::trans(x - mu) * inv(sigma) * (x - mu));
  double expTerm;
  expTerm = AexpTerm(0, 0);

  out = rootTerm * expTerm;
  return(log(out));
}

//' Random Walk Metropolis-Hastings MCMC
//'
//' @param target the R function that must return the log likelihood of the target distribution we want to sample from
//' @param init_theta the vector of initial parameter values
//' @param covmat the covariance matrix of the proposal distribution
//' @param n_iterations the integer value, how long to run the chain
//' @return R list: theta_trace, acceptance_rate
//' @examples
//' library(mcmcSampler)
//' p.log <- function(x) {
//'   B <- 0.03 # controls 'bananacity'
//'   -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2
//' }
//' sample <- rwMCMC(target=p.log, init_theta=c(10,10), covmat=diag(c(1,1)),n_iterations=1e4)
// [[Rcpp::export]]
List rwMCMC(Function target, arma::vec init_theta, arma::mat covmat, int n_iterations){

  std::default_random_engine engine;
  std::uniform_real_distribution<> uniform(0.0,1.0);
  std::normal_distribution<> normal(0.0,1.0);

  arma::vec theta_current = init_theta;
  arma::vec theta_propose = init_theta;
  arma::mat covmat_proposal = covmat;

  arma::mat theta_trace = arma::zeros(n_iterations,init_theta.n_elem);
  arma::vec target_trace(n_iterations);

  //evaluate target distribution at theta_init
  double target_theta_current; //evaluate target at current theta
  target_theta_current = as<double>(wrap(target(theta_current)));

  double acceptance_rate = 0.0;
  arma::vec acceptance_trace(n_iterations);

  //mcmc loop
  for(int i=1; i<=n_iterations; i++){

    //propose new theta
    arma::rowvec Y(covmat_proposal.n_cols);
    for(int i = 0; i < covmat_proposal.n_cols; i++){
      Y(i) = normal(engine);
    }
    arma::rowvec theta_proposeNew = theta_current.t() + Y * arma::chol(covmat_proposal);
    theta_propose = theta_proposeNew.t();

    //evaluate target distribution at proposed theta
    double target_theta_propose;
    target_theta_propose = as<double>(wrap(target(theta_propose)));
    bool is_accepted;
    double log_acceptance;

    //if posterior is 0 immediately reject
    if(!std::isfinite(target_theta_propose)){
      is_accepted = false;
      log_acceptance = -std::numeric_limits<double>::infinity();
    } else {
      //compute Metropolis-Hastings accept ratio?
        log_acceptance = target_theta_propose - target_theta_current;
        log_acceptance = log_acceptance + mvrnorm_pdff(theta_current, theta_propose, covmat_proposal);
        log_acceptance = log_acceptance - mvrnorm_pdff(theta_propose, theta_current, covmat_proposal);
    }

    //evaluate acceptance probability
    double A = uniform(engine);
    if(log(A) < log_acceptance){
      //accept proposed parameter set
      is_accepted = true;
      theta_current = theta_propose;
      target_theta_current = target_theta_propose;
    } else {
      is_accepted = false;
    }

    //store trace of MCMC
    theta_trace.row(i-1) = theta_current.t();
    target_trace(i-1) = target_theta_current;

    //update acceptance_rate
    if(i == 1){
      acceptance_rate = is_accepted;
    } else {
      acceptance_rate = acceptance_rate + (is_accepted - acceptance_rate) / i;
    }
    acceptance_trace(i-1) = acceptance_rate;

  }

  return(List::create(Named("samples")=theta_trace,
                      Named("log.p")=target_trace,
                      Named("acceptance_rate")=acceptance_trace));
}
