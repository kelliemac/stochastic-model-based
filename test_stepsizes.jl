#-----------------------------------------------------------------------------------
# Tuning the constant stepsize for SGD and SMD
#-----------------------------------------------------------------------------------

using Random
using PyPlot
using LaTeXStrings, Printf

include("solve_cov_est.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 10000;
r = 2;  # rank
d = 100;
stoch_err = 0.01;  # standard deviation of errors in stochastic measurements b
init_radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

subgrad_stepsizes = [1.25, 1.5, 1.75] * 1e-4;
mirror_stepsizes = [0.5, 0.6, 0.7];
step_colors = ["#1f78b4", "#33a02c", "red"]; #, "blue"];

Random.seed!(321);  # for reproducibility

#-------------------------------------
#   Initialize plots
#-------------------------------------
subgrad_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SGD for covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-4, 1e1)

mirror_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SMD for covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-4, 1e1)

#-----------------------------------------------------------------------------------------
#   Run stochastic subgradient method for each dimension
#-----------------------------------------------------------------------------------------
for i=1:length(step_colors)
    ηsubgrad = subgrad_stepsizes[i];
    ηmirror = mirror_stepsizes[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    pert = randn(d,r);
    Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Run the two methods
    (subgrad_err_hist, ~) = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter,
                                                    fill(ηsubgrad, maxIter), method="subgradient")
    (mirror_err_hist, ~) = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter,
                                                    fill(ηmirror, maxIter), method="mirror")

    # Add errors to plots
    plt[:figure](subgrad_plot[:number])
    semilogy(subgrad_err_hist, color=step_colors[i], label=@sprintf("step=%1.2e", ηsubgrad));

    plt[:figure](mirror_plot[:number])
    semilogy(mirror_err_hist, color=step_colors[i], label=@sprintf("step=%1.2e", ηmirror));
end

# Add legends to plots and save final plots
plt[:figure](subgrad_plot[:number])
legend(loc="upper right")
savefig("plots/sgd_stepsize_test.pdf");

plt[:figure](mirror_plot[:number])
legend(loc="upper right")
savefig("plots/smd_stepsize_test.pdf");
