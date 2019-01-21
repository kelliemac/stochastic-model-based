#-----------------------------------------------------------------------------------
# Testing SGD and SMD, clipped and non-clipped models
# Step size agnostic plots: covergence at range of stepsizes
#-----------------------------------------------------------------------------------

using Random, PyPlot, LaTeXStrings, Printf
include("solve_cov_est.jl");

Random.seed!(321);  # for reproducibility

#-------------------------------------------------------------
#    What two methods are we comparing?
#-------------------------------------------------------------
method1 = "subgradient"
method2 = "mirror"
clipped1 = false
clipped2 = true

#-------------------------------------
#    Set parameters
#-------------------------------------
maxIter = 9000;
r = 10;  # rank
d = 100;
stoch_err = 0.01;  # standard deviation of errors in stochastic measurements b
init_radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

stepSizes = 10.0 .^(range(-6,2,length=17))
num_rounds = 10;

#-----------------------------------------------------------
#   Initialize vectors to track final errors
#-----------------------------------------------------------
dist_errors_1 = fill(NaN, length(stepSizes));
fun_errors_1 = fill(NaN, length(stepSizes));
dist_errors_2 = fill(NaN, length(stepSizes));
fun_errors_2 = fill(NaN, length(stepSizes));

#-----------------------------------------------------------------------------------------
#   Run the two methods for each choice of stepsize
#-----------------------------------------------------------------------------------------
for i=1:length(stepSizes)
    η = stepSizes[i];

    dist_errors_tally_1 = 0;
    fun_errors_tally_1 = 0;
    dist_errors_tally_2 = 0;
    fun_errors_tally_2 = 0;

    for j=1:num_rounds
        # Generate true matrix and initialization
        Xtrue = randn(d,r);
        pert = randn(d,r);
        Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

        # Run the two methods and record final errors
        @printf("Running Method 1 for stepsize %1.2e (Round %i of %i)\n", η, 2*num_rounds*(i-1)+2*j-1, 2*length(stepSizes)*num_rounds);
        (err_hist_1, fun_hist_1) = solve_cov_est(Xinit, Xtrue, stoch_err, maxIter,
                                                       fill(η, maxIter), method=method1, clipped=clipped1, verbose=false)
        dist_errors_tally_1 += err_hist_1[end];
        fun_errors_tally_1 += fun_hist_1[end];

        @printf("Running Method 2 for stepsize %1.2e (Round %i of %i)\n", η, 2*num_rounds*(i-1)+2*j, 2*length(stepSizes)*num_rounds);
        (err_hist_2, fun_hist_2)  = solve_cov_est(Xinit, Xtrue, stoch_err, maxIter,
                                                        fill(η, maxIter), method=method2, clipped=clipped2, verbose=false)
        dist_errors_tally_2 += err_hist_2[end];
        fun_errors_tally_2 += fun_hist_2[end];

    end

    dist_errors_1[i] = dist_errors_tally_1 / num_rounds;
    fun_errors_1[i] = fun_errors_tally_1 /num_rounds;

    dist_errors_2[i] = dist_errors_tally_2 / num_rounds;
    fun_errors_2[i] = fun_errors_tally_2 / num_rounds;

end

distance_plot = figure(figsize=[10,6]);
title(@sprintf("Relative Distance Errors (r=%i, d=%i, %i iterations)", r,d,maxIter));
xlabel(L"Stepsize $\eta$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
loglog(stepSizes, dist_errors_1, label=string(method1, clipped1 ? " clipped" : "") );
loglog(stepSizes, dist_errors_2, label=string(method2, clipped2 ? " clipped" : ""));
ylim(1e-5,1e3)
legend(loc="upper right")
savefig("plots/agnostic_cov_est_distances.pdf");

fun_plot = figure(figsize=[10,6]);
title(@sprintf("Absolute Function Errors (r=%i, d=%i, %i iterations)", r,d,maxIter));
xlabel(L"Stepsize $\eta$");
ylabel(L"Empirical Function Error $\quad \hat f (X_{final}) - \hat f (X_{true})$");
loglog(stepSizes, fun_errors_1, label=string(method1, clipped1 ? " clipped" : "") );
loglog(stepSizes, fun_errors_2, label=string(method2, clipped2 ? " clipped" : "") );
ylim(1e-5,1e10)
legend(loc="upper right")
savefig("plots/agnostic_cov_est_values.pdf");
