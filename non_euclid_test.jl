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

include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 1000;
r = 2;  # rank
dims = [10]; #, 100];
dims_colors = ["#1f78b4"]; #, "#33a02c"];
stoch_err = 0.0;  # standard deviation of errors in stochastic measurements b

Random.seed!(123);  # for reproducibility

#-------------------------------------
#   Initialize plot
#-------------------------------------
figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"$\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SGD and SMD for covariance estimation (r=%i)", r));
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

    α = 100.0;  # any positive number
    η = 1 / ( 1/λ + 1/α * sqrt(maxIter+1) );  # constant step size
    stepSizes = fill(η, maxIter);

    # Run subgradient method
    err_hist_subgrad = solve_cov_est_subgradient(Xinit, Xtrue, stepSizes, maxIter, stoch_err)
    err_hist_mirror = solve_cov_est_mirror(Xinit, Xtrue, stepSizes, maxIter)

    # Add errors for this dimension to plot
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legend to plot and save final results for all dimensions
legend(loc="lower right")
savefig("non_eucl_test.pdf");
