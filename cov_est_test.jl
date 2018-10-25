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
using Random  # for setting the random seed
using LaTeXStrings  # for latex in plot labels
using Printf  # for printing parameter values in plot labels
using PyPlot

include("solve_cov_est_mirror.jl");
include("solve_cov_est_subgradient.jl");

#-------------------------------------
#   Set parameters
#-------------------------------------
maxIter = 1000;
r = 2;  # rank
dims = [10, 100];
dims_colors = ["#1f78b4", "#33a02c"];
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b

Random.seed!(123);  # for reproducibility

#-------------------------------------------------------------------------------------------
#   Initialize plots - distances to solution and function values
#-------------------------------------------------------------------------------------------
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
#   Run SGD and SMD method for each dimension
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
    stepSizes_subgrad = fill(3e-4, maxIter);
    stepSizes_mirror = fill(3e-4, maxIter);

    # Run SGD and SMD
    (err_hist_subgrad, fun_hist_subgrad) = solve_cov_est_subgradient(Xinit, Xtrue, stepSizes_subgrad, maxIter, stoch_err)
    (err_hist_mirror, fun_hist_mirror) = solve_cov_est_mirror(Xinit, Xtrue, stepSizes_mirror, maxIter, stoch_err)

    # Plot distances to solution
    plt[:figure](dist_fig[:number])
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));

    # Plot function values
    plt[:figure](funval_fig[:number])
    semilogy(fun_hist_subgrad, linestyle="--", color=dims_colors[i], label=@sprintf("SGD, d=%i", d));
    semilogy(fun_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legends to plots and save the final composite plots
plt[:figure](dist_fig[:number])
legend(loc="lower right")
savefig("cov_est_distances.pdf");

plt[:figure](funval_fig[:number])
legend(loc="lower right")
savefig("cov_est_function_values.pdf");
