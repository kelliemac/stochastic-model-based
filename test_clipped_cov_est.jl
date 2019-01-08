#-----------------------------------------------------------------------------------
# Testing SGD and clipped SMD, step size agnostic
#-----------------------------------------------------------------------------------

using Random
using PyPlot
using LaTeXStrings, Printf

include("solve_cov_est.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 100;
r = 2;  # rank
d = 100;
stoch_err = 0.01;  # standard deviation of errors in stochastic measurements b
init_radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

stepSizes = [1e-4, 1e-5, 1e-6];

Random.seed!(321);  # for reproducibility

#-----------------------------------------------------------
#   Initialize vectors to track final errors
#-----------------------------------------------------------
dist_errors_sgd = fill(NaN, length(stepSizes));
dist_errors_smd = fill(NaN, length(stepSizes));

fun_errors_sgd = fill(NaN, length(stepSizes));
fun_errors_smd = fill(NaN, length(stepSizes));

#-----------------------------------------------------------------------------------------
#   Run the two methods for each choice of stepsize
#-----------------------------------------------------------------------------------------
for i=1:length(stepSizes)
    η = stepSizes[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    pert = randn(d,r);
    Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Run the two methods
    (sgd_err_hist, sgd_fun_hist) = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter,
                                                   fill(η, maxIter), method="subgradient", clipped=false)
    (mirror_err_hist, mirror_fun_hist)  = solve_cov_est(Xinit, Xtrue, stoch_err,  maxIter,
                                                    fill(η, maxIter), method="mirror", clipped=true)

    # Record final errors
    dist_errors_sgd[i] = sgd_err_hist[end];
    dist_errors_smd[i] = sgd_fun_hist[end];

    fun_errors_sgd[i] = mirror_err_hist[end];
    fun_errors_smd[i] = mirror_fun_hist[end];
end

distance_plot = figure(figsize=[10,6]);
title(@sprintf("Relative Distance Errors for SGD and Clipped SMD (r=%i, d=%i)", r,d));
xlabel(L"Stepsize $\eta$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
# xlim(0,maxIter)
# ylim(1e-4, 1e1)
semilogy(stepSizes, dist_errors_sgd, label="sgd");
semilogy(stepSizes, dist_errors_smd, label="clipped smd");
legend(loc="upper right")
savefig("plots/clipped_test_distances.pdf");

fun_plot = figure(figsize=[10,6]);
title(@sprintf("Absolute Function Errors for SGD and Clipped SMD (r=%i, d=%i)", r,d));
xlabel(L"Stepsize $\eta$");
ylabel(L"Empirical Function Error $\quad \hat f (X_{final}) - \hat f (X_{true})$");
# xlim(0,maxIter)
# ylim(1e-4, 1e1)
semilogy(stepSizes, fun_errors_sgd, label="sgd");
semilogy(stepSizes, fun_errors_smd, label="clipped smd");
legend(loc="upper right")
savefig("plots/clipped_test_values.pdf");
