#   Find the minimum number of iterations (min taken over choices of stepsizes)
#   that is required to achieved accuracy ϵ (in function values)
#   when applying the subgradient method to covariance estimation.

#   Average results over a few trials before extracting number of iterations, due to
#   stochastic variability.

using Random, Printf, PyPlot
include("solve_enet.jl");

Random.seed!(123);  # for reproducibility

# Set problem parameters
r = 5;  # rank
d = 100;
stoch_err = sqrt(10);  # standard deviation of errors in stochastic measurements b
maxIter = 10000;
init_radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

# Set accuracy parameters
ϵ = 1e-3;
num_trials = 1;
subgrad_stepsizes = [1e-6, 5e-6];

# Make plot
figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
# ylabel("Empirical Function Value Error");
ylabel("Normalized distance to solution set");
title(@sprintf("SGD for e-net covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-4, 1e1)

@printf("------------------------------------------------------------------\n")
@printf("Num. iterations til accuracy %1.2e (emp. function values) \n", ϵ)
@printf("------------------------------------------------------------------\n")

for η in subgrad_stepsizes

    sum_fun_hists = fill(0, maxIter);
    sum_err_hists = fill(0, maxIter);

    for i in 1:num_trials
        # Generate true matrix we are searching for
        Xtrue = randn(d,r);

        # Generate initial point
        pert = randn(d,r);
        Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

        # Run the method and save progress
        steps = fill(η, maxIter);
        (err_hist, fun_hist) = solve_enet(Xinit, Xtrue, stoch_err,  maxIter, steps,
                                                                    method="subgradient", verbose=false);
        sum_fun_hists += fun_hist;
        sum_err_hists += err_hist;
    end

    # avg_fun_hist = sum_fun_hists / num_trials;
    # semilogy(avg_fun_hist, label=@sprintf("step size: %1.2e", η) )
    avg_err_hist = sum_err_hists / num_trials;
    semilogy(avg_err_hist, label=@sprintf("step size: %1.2e", η) )

    iters_to_accuracy = findfirst( x -> (x < ϵ), avg_err_hist );
    if typeof(iters_to_accuracy)==Nothing
        iters_to_accuracy = NaN;
    end

    @printf("stepsize: %1.2e, final accuracy: %1.2e, iters to desired accuracy: %i \n",
                        η, avg_err_hist[end], iters_to_accuracy)

end

legend(loc="upper right")
savefig("plots/finding_enet_num_iters.pdf");
