#------------------------------------------------------------------------------------------------------------
#   Code to solve
#   min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   via
#   1) stochastic subgradient method (Φ = 1/2 ||⋅||^2)
#   2) stochastic mirror descent (Φ = a quartic polynomial)
#
#   Stochastic subgradients are
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X
#   where (a, b) are chosen randomly.
#
# Saves a plot of the relative errors:  min_{U orthogonal} || X U - Xtrue ||₂² / || Xtrue ||₂²
#
#------------------------------------------------------------------------------------------------------------
using Random
using LaTeXStrings
using Printf
using PyPlot

include("solve_cov_est_mirror.jl");
include("solve_cov_est_subgradient.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 1000;
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
title(@sprintf("Distance to solution for covariance estimation (r=%i)", r));
xlim(0,maxIter)

funval_fig = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Empirical Function Value");
title(@sprintf("Function values for covariance estimation (r=%i)", r));
xlim(0,maxIter)

#-----------------------------------------------------------------------------------------
#   Run stochastic subgradient method for each dimension
#-----------------------------------------------------------------------------------------
for i in 1:length(dims)
    d = dims[i]

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
    pert = randn(d,r);
    Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Specify step sizes
    τ = 10.0;
    ρ = 0.0;  # linear models are convex
    λ = 0.9 / (τ+ρ);

    α = 0.01;  # any positive number
    η = 1 / ( 1/λ + 1/α * sqrt(maxIter+1) );  # constant step size
    stepSizes_subgrad = fill(η, maxIter);

    α = 0.01;  # any positive number
    η = 1 / ( 1/λ + 1/α * sqrt(maxIter+1) );  # constant step size
    stepSizes_mirror = fill(η, maxIter);

    # Run subgradient method
    (err_hist_subgrad, fun_hist_subgrad) = solve_cov_est_subgradient(Xinit, Xtrue, stepSizes_subgrad, maxIter, stoch_err)
    (err_hist_mirror, fun_hist_mirror) = solve_cov_est_mirror(Xinit, Xtrue, stepSizes_mirror, maxIter, stoch_err)

    # Add errors for this dimension to plot
    plt[:figure](dist_fig[:number])
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));

    # Add function values for this dimension to plot
    plt[:figure](funval_fig[:number])
    semilogy(fun_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(fun_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legend to plot and save final results for all dimensions
plt[:figure](dist_fig[:number])
legend(loc="lower right")
savefig("cov_est_distances.pdf");

plt[:figure](funval_fig[:number])
legend(loc="lower right")
savefig("cov_est_function_values.pdf");
