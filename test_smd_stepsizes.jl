#-----------------------------------------------------------------------------------
# Tuning the constant stepsize for SMD.
#-----------------------------------------------------------------------------------

using Random
using LaTeXStrings
using Printf
using PyPlot

include("solve_cov_est_mirror.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 2500;
r = 2;  # rank
d = 10;
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b

# stepsizes = [0.1, 1.0, 5.0, 10.0];  # for d=100
stepsizes = [0.2, 0.3, 0.4, 0.5];  # for d=10
step_colors = ["#1f78b4", "#33a02c", "red", "blue"];

Random.seed!(321);  # for reproducibility

#-------------------------------------
#   Initialize plot
#-------------------------------------
figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"$\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SMD for covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)

#-----------------------------------------------------------------------------------------
#   Run stochastic subgradient method for each dimension
#-----------------------------------------------------------------------------------------
for i=1:length(stepsizes)
    η = stepsizes[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
    pert = randn(d,r);
    Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    stepSizes = fill(η, maxIter);

    # Run subgradient method
    (err_hist,~) = solve_cov_est_mirror(Xinit, Xtrue, stepSizes, maxIter, stoch_err)

    # Add errors for this stepsize to plot
    semilogy(err_hist, color=step_colors[i], label=@sprintf("step=%1.2e", η));
end

# Add legend to plot and save final results for all dimensions
legend(loc="upper right")
savefig("smd_stepsize_test.pdf");
