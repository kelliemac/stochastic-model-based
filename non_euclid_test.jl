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
# Saves a plot of the relative errors:  || X - Xtrue ||₂² / || Xtrue ||₂²
#
#------------------------------------------------------------------------------------------------------------
using Random
using LaTeXStrings
using Printf
using PyPlot
using ColorBrewer

include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 1000;
dims = [10, 100];
dims_colors = ["#1f78b4", "#33a02c"];
r = 2;  # rank

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
    radius = 1.0;
    pert = randn(d,r);
    X_init = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Specify step sizes
    stepSizes = fill(0.01, maxIter) / d ;

    # Run subgradient method
    err_hist_subgrad = solve_cov_est_subgradient(X_init, Xtrue, stepSizes, maxIter)
    err_hist_mirror = solve_cov_est_mirror(X_init, Xtrue, stepSizes, maxIter)

    # Add errors for this dimension to plot
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legend to plot and save final results for all dimensions
legend(loc="lower right")
savefig("non_eucl_test.pdf");
