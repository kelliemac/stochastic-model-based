#   Find the minimum number of iterations (min taken over choices of stepsizes)
#   that is required to achieved accuracy ϵ (in function values)
#   when applying the subgradient method to covariance estimation.

#   Average results over a few trials before extracting number of iterations, due to
#   stochastic variability.

using Printf
include("solve_cov_est.jl");

Random.seed!(123);  # for reproducibility

# Set problem parameters
r = 2;  # rank
d = 100;
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b
maxIter = 3000;

# Set accuracy parameters
ϵ = 1e-1;
num_trials = 5;
subgrad_stepsizes = [6.0, 9.0, 12.0, 15.0] * 1e-5;  # for d=100

@printf("------------------------------------------------------------------\n")
@printf("Num. iterations til accuracy %1.2e (emp. function values) \n", ϵ)
@printf("------------------------------------------------------------------\n")

for η in subgrad_stepsizes

    sum_fun_hists = fill(0, maxIter);

    for i in 1:num_trials
        # Generate true matrix we are searching for
        Xtrue = randn(d,r);

        # Generate initial point
        radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
        pert = randn(d,r);
        Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

        # Run the method and save progress
        (err_hist, fun_hist) = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter, fill(η, maxIter),
                                                                    method="subgradient", verbose=false);
        sum_fun_hists += fun_hist;
    end

    avg_fun_hist = sum_fun_hists / num_trials;
    iters_to_accuracy = findfirst( x -> (x < ϵ), avg_fun_hist );
    if typeof(iters_to_accuracy)==Nothing
        iters_to_accuracy = NaN;
    end

    @printf("stepsize: %1.2e, final accuracy: %1.2e, iters to desired accuracy: %i \n", η, avg_fun_hist[end], iters_to_accuracy)

end
