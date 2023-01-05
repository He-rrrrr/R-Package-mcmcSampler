#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::plugins(cpp11)]]
#include <random>
using namespace Rcpp;



//更新正态分布的协方差矩阵
arma::mat update_sigma(arma::mat sigma, arma::vec residual, double i) {
	arma::mat out = (sigma * (i - 1) + (i - 1) / i * residual * trans(residual)) / i;
	return(out);
}

//多元正态分布密度
double mvrnorm_pdf(arma::vec x, arma::vec mu, arma::mat sigma) {

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



//' Metropolis-Hastings MCMC with Adaptive Proposal Kernel
//'
//' @param target the a R function that must return the log likelihood of the target distribution we want to sample from
//' @param init_theta a vector of initial parameter values
//' @param covmat the covariance matrix of the proposal distribution
//' @param n_iterations the integer value, how long to run the chain
//' @param adapt_size_start the number of accepted jumps after which to begin adapting size of proposal covariance matrix
//' @param info the print information on chain every X n_iterations
//' @param seedMH the seed of RNG
//' @param eps the a constant that we may choose very small compared to the size of S
//' @return R list: theta_trace,sigma_empirical,acceptance_rate
//' @examples
//' library(mcmcSampler)
//' p.log <- function(x) {
//' B <- 0.03 # controls 'bananacity'
//' -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2
//' }
//' set.seed(123)
//'  banana1 <- adaptMCMC(target=p.log,init_theta=c(10,10),covmat=diag(c(1,1)),n_iterations=1e3,adapt_size_start=10,acceptance_rate_weight=0,acceptance_window=0,info=1e2, seedMH = 123)
//'
//' par(mfrow=c(1,2))
//' x1 <- seq(-15, 15, length=100)
//' x2 <- seq(-15, 15, length=100)
//' d.banana <- matrix(apply(expand.grid(x1, x2), 1, p.log), nrow=100)
//' image(x1, x2, exp(d.banana), col=cm.colors(60))
//' contour(x1, x2, exp(d.banana), add=TRUE, col=gray(0.6))
//' lines(banana1$theta_trace, type='l')
//' matplot(banana1$theta_trace,type="l")
// [[Rcpp::export]]
List adaptMCMC(Function target, arma::vec init_theta, arma::mat covmat, int n_iterations, int adapt_size_start, double acceptance_rate_weight, int acceptance_window,
    int info, double seedMH = 1, double eps = 0.05) {

    std::mt19937 engine(seedMH); //set seed of uniform RNG
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    std::normal_distribution<> normal(0.0, 1.0);


    //初始化参数和建议分布的协方差矩阵?
    arma::vec theta_current = init_theta;
    arma::vec theta_propose = init_theta;
    arma::mat covmat_proposal = covmat;

    bool adapting_size = false;

    arma::mat theta_trace = arma::zeros(n_iterations, init_theta.n_elem);//?????켣??n_iterations??????
    arma::cube sigma_empirical = arma::zeros(covmat.n_rows, covmat.n_cols, n_iterations);//n_iterations??Э????????

    //计算theta_init在target中的值ֵ
    double target_theta_current; //evaluate target at current theta
    target_theta_current = as<double>(wrap(target(theta_current)));

    //初始接受率
    double acceptance_rate = 0.234;
    arma::vec acceptance_series;


    arma::mat covmat_empirical = covmat;
    double scaling_sd = 1;

    arma::vec theta_mean = theta_current;

    //main mcmc loop
    for (int i = 1; i <= n_iterations; i++) {

        //adaptive routine 从t0开始更新建议分布的协方差矩阵
        if (adapt_size_start != 0 && i >= adapt_size_start) {
            if (!adapting_size) {
                Rcout << "Begin adapting size of sigma at iter: " << i << std::endl;
                adapting_size = true;
            }
            //给定sd
            scaling_sd = pow(2.38, 2) / init_theta.n_elem;

            //更新建议分布的协方差矩阵
            arma::mat I_d = arma::eye(covmat.n_rows, covmat.n_cols);
            arma::mat covmat_proposal_new = scaling_sd * covmat_empirical + scaling_sd * eps * I_d;
            if (!any(covmat_proposal_new.diag() < 2E-16)) {
                covmat_proposal = covmat_proposal_new;
            }
        }
        //print chain diagnostics
        if (i % info == 0) {
            Rcout << "At iter: " << i << ", acceptance rate is: " << acceptance_rate << std::endl;
        }

        //在建议分布中抽取新的theta
        arma::rowvec Y(covmat_proposal.n_cols);
        for (int i = 0; i < covmat_proposal.n_cols; i++) {
            Y(i) = normal(engine);
        }
        arma::rowvec theta_proposeNew = theta_current.t() + Y * arma::chol(covmat_proposal);
        theta_propose = theta_proposeNew.t();

        //计算新theta的接受率
        //evaluate target distribution at proposed theta
        double target_theta_propose;
        target_theta_propose = as<double>(wrap(target(theta_propose)));
        bool is_accepted;
        double log_acceptance;

        //if posterior is 0 immediately reject
        if (!std::isfinite(target_theta_propose)) {
            is_accepted = false;
            log_acceptance = -std::numeric_limits<double>::infinity();
        }
        else {
            //compute Metropolis-Hastings ratio (acceptance probability)
            log_acceptance = target_theta_propose - target_theta_current;
            log_acceptance = log_acceptance + mvrnorm_pdf(theta_current, theta_propose, covmat_proposal);
            log_acceptance = log_acceptance - mvrnorm_pdf(theta_propose, theta_current, covmat_proposal);
        }

        //以接受率维概率接收新theta
        double A = uniform(engine);
        if (log(A) < log_acceptance) {
            //accept proposed parameter set
            is_accepted = true;
            theta_current = theta_propose;
            target_theta_current = target_theta_propose;
        }
        else {
            is_accepted = false;
        }

        //store trace of MCMC
        theta_trace.row(i - 1) = theta_current.t();

        //update acceptance rate
        if (i == 1) {
            acceptance_rate = is_accepted;
        }
        else {
            if (acceptance_rate_weight == 0) {
                if (acceptance_window == 0) {
                    acceptance_rate = acceptance_rate + (is_accepted - acceptance_rate) / i;
                }
                else {
                    arma::vec is_accepted_vec(1);
                    is_accepted_vec(0) = is_accepted;
                    acceptance_series = arma::join_cols<arma::mat>(is_accepted_vec, acceptance_series);
                    if (acceptance_series.n_elem > acceptance_window) {
                        int series_length = acceptance_series.n_elem;
                        acceptance_series.resize(series_length - 1);
                    }
                    acceptance_rate = arma::mean(acceptance_series);
                }
            }
            else {
                acceptance_rate = acceptance_rate * (1 - acceptance_rate_weight) + is_accepted * acceptance_rate_weight;
            }
        }

        //update empirical covariance matrix (estimation of sigma)
        arma::vec residual = theta_current - theta_mean;
        covmat_empirical = update_sigma(covmat_empirical, residual, i);///最新的协方差矩阵
        theta_mean = theta_mean + residual / i;

        sigma_empirical.slice(i - 1) = covmat_empirical;

    }

    return(List::create(Named("theta_trace") = theta_trace, Named("sigma_empirical") = sigma_empirical, Named("acceptance_rate") = acceptance_rate));
}




// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automaticallyrun after the compilation.
//

