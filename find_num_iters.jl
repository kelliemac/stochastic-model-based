#   Find the minimum number of iterations (min taken over choices of stepsizes)
#   that is required to achieved accuracy ϵ (in function values)
#   when applying the subgradient method to covariance estimation.

#   Average results over a few trials before extracting number of iterations, due to
#   stochastic variability.

using Random, Printf, PyPlot
include("solve_cov_est.jl");

Random.seed!(123);  # for reproducibility

# Set problem parameters
r = 2;  # rank
d = 100;
stoch_err = 0.0;  # standard deviation of errors in stochastic measurements b
maxIter = 20;
init_radius = 0.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

# Set accuracy parameters
ϵ = 1e-3;
num_trials = 5;
subgrad_stepsizes = [1e-4, 5e-5, 1e-5, 5e-6];

# Make plot
figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Relative Empirical Function Value Error");
title(@sprintf("SGD for covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-3, 1e2)

@printf("------------------------------------------------------------------\n")
@printf("Num. iterations til accuracy %1.2e (emp. function values) \n", ϵ)
@printf("------------------------------------------------------------------\n")

for η in subgrad_stepsizes

    sum_fun_hists = fill(0, maxIter);

    for i in 1:num_trials
        # Generate true matrix we are searching for
        Xtrue = randn(d,r);

        # Generate initial point
        pert = randn(d,r);
        Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

        # Run the method and save progress
        steps = fill(η, maxIter);
        (err_hist, fun_hist) = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter, steps,
                                                                    method="subgradient", verbose=false);
        sum_fun_hists += fun_hist;
    end

    avg_fun_hist = sum_fun_hists / num_trials;
    semilogy(avg_fun_hist, label=@sprintf("step size: %1.2e", η) )

    iters_to_accuracy = findfirst( x -> (x < ϵ), avg_fun_hist );
    if typeof(iters_to_accuracy)==Nothing
        iters_to_accuracy = NaN;
    end

    @printf("stepsize: %1.2e, final accuracy: %1.2e, iters to desired accuracy: %i \n", η, avg_fun_hist[end], iters_to_accuracy)

end

legend(loc="lower left")
savefig("plots/finding_num_iters.pdf");
