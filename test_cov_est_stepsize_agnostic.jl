#------------------------------------------------------------------------------------------------------------
#   Solving
#   min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   via
#   1) stochastic subgradient method (Φ = 1/2 ||⋅||^2)
#   2) stochastic mirror descent (Φ = a quartic polynomial)
#
#   Note that stochastic subgradients are
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X
#   where (a, b) are chosen randomly.
#
# Saves plots of 1) relative errors:  min_{U orthogonal} || X U - Xtrue ||₂² / || Xtrue ||₂²
#                        2) empirical function values
#------------------------------------------------------------------------------------------------------------
using Random  # for setting the random seed
using PyPlot
using LaTeXStrings, Printf  # for plot labels

include("solve_cov_est.jl");

#-------------------------------------
#   Set parameters
#-------------------------------------
r = 2;  # rank
d = 100;  # ambient dimension
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b

maxIter = 5000;
steps = 10.0 .^ [-5, -4, -3, -2, -1, 0, 1, 2];
num_trials = 10;

Random.seed!(123);  # for reproducibility

#-------------------------------------------------------------------------------------------
#   For plots - distances to solution and function values
#-------------------------------------------------------------------------------------------
distances_subgrad = fill(NaN, length(steps));
distances_mirror = fill(NaN, length(steps));

vals_subgrad = fill(NaN, length(steps));
vals_mirror = fill(NaN, length(steps));

#-----------------------------------------------------------------------------------------
#   Run SGD and SMD method for each dimension
#-----------------------------------------------------------------------------------------
for i in 1:length(steps)
    η = steps[i];

    sum_distances_subgrad = 0;
    sum_distances_mirror = 0;
    sum_vals_subgrad = 0;
    sum_vals_mirror = 0;

    for t in 1:num_trials
        # Generate true matrix we are searching for
        Xtrue = randn(d,r);

        # Generate initial point
        radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
        pert = randn(d,r);
        Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

        # Specify step sizes
        stepSizes = fill(η, maxIter);

        # Run SGD and SMD
        (err_hist_subgrad, fun_hist_subgrad) = solve_cov_est(Xinit, Xtrue, stoch_err,
                                                                          maxIter, stepSizes, method="subgradient")
        (err_hist_mirror, fun_hist_mirror) = solve_cov_est(Xinit, Xtrue, stoch_err,
                                                                          maxIter, stepSizes, method="mirror")

        # Record final errors
        sum_distances_subgrad += err_hist_subgrad[end];
        sum_distances_mirror += err_hist_mirror[end];

        sum_vals_subgrad += fun_hist_subgrad[end];
        sum_vals_mirror += fun_hist_mirror[end];
    end

    distances_subgrad[i] = sum_distances_subgrad / num_trials;
    distances_mirror[i] = sum_distances_mirror / num_trials;

    vals_subgrad[i] = sum_vals_subgrad / num_trials;
    vals_mirror[i] = sum_vals_mirror / num_trials;
end

# Make and save distance plot
dist_fig = figure(figsize=[10,6]);
xlabel(L"Step Size $\eta$");
ylabel(L"Average Distance $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("Distance to solution for covariance estimation (r=%i, d=%i)", r, d));
xlim(minimum(steps),maximum(steps))

loglog(steps, distances_subgrad, label="SGD")
loglog(steps, distances_mirror, label="SMD")
legend(loc="upper right")
savefig("plots/cov_est_average_distances.pdf");

# Make and save function values plot
fun_vals_fig = figure(figsize=[10,6]);
xlabel(L"Step Size $\eta$");
ylabel("Average final function value");
title(@sprintf("Function values for covariance estimation (r=%i, d=%i)", r, d));
xlim(minimum(steps),maximum(steps))

loglog(steps, vals_subgrad, label="SGD")
loglog(steps, vals_mirror, label="SMD")
legend(loc="upper right")
savefig("plots/cov_est_average_values.pdf");
