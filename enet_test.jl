#------------------------------------------------------------------------------------------------------------
#   Code to solve elastic net-ish version of covariance estimation
#   min_{X \in \R^{d x r}} F(X) :=  E_{a,b} | <XX^T, aa^T> - b | + (<XX^T, aa^T> - b )^2
#
#------------------------------------------------------------------------------------------------------------
using Random
using LaTeXStrings
using Printf
using PyPlot

include("solve_enet_subgradient.jl");
include("solve_enet_mirror.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 100;
r = 2;  # rank
dims = [10, 100];
dims_colors = ["#1f78b4", "#33a02c"];
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b

Random.seed!(123);  # for reproducibility

#-------------------------------------
#   Initialize plot
#-------------------------------------
dist_fig = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"$\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("Distance to solution for elastic net covariance estimation (r=%i)", r));
xlim(0,maxIter)

funval_fig = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Empirical Function Value");
title(@sprintf("Function values for elastic net covariance estimation (r=%i)", r));
xlim(0,maxIter)

#-----------------------------------------------------------------------------------------
#   Run stochastic subgradient method for each dimension
#-----------------------------------------------------------------------------------------
for i in 1:length(dims)
    d = dims[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
    pert = randn(d,r);
    Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Specify step sizes
    stepSizes_subgrad = fill(1e-4, maxIter);
    stepSizes_mirror = fill(10.0, maxIter);

    # Run subgradient method
    (err_hist_subgrad, fun_hist_subgrad) = solve_enet_subgradient(Xinit, Xtrue, stepSizes_subgrad, maxIter, stoch_err)
    # (err_hist_mirror, fun_hist_mirror) = solve_enet_mirror(Xinit, Xtrue, stepSizes_mirror, maxIter, stoch_err)

    # Add errors for this dimension to plot
    plt[:figure](dist_fig[:number])
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    # semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));

    # Add function values for this dimension to plot
    plt[:figure](funval_fig[:number])
    semilogy(fun_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    # semilogy(fun_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legend to plot and save final results for all dimensions
plt[:figure](dist_fig[:number])
legend(loc="lower right")
savefig("enet_test_distances.pdf");

plt[:figure](funval_fig[:number])
legend(loc="lower right")
savefig("enet_test_values.pdf");
