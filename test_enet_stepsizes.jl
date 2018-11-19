#--------------------------------------------------------------------------------------------------
# Tuning the constant stepsize for SGD and SMD - E-NET VERSION
#--------------------------------------------------------------------------------------------------

using Random
using PyPlot
using LaTeXStrings, Printf

include("solve_enet.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 7000;
r = 5;  # rank
d = 100;
stoch_err = sqrt(10);  # standard deviation of errors in stochastic measurements b
init_radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit

num_tests = 5;
subgrad_stepsizes = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6];
mirror_stepsizes = [0.3, 0.4, 0.5, 0.6, 0.7];

Random.seed!(321);  # for reproducibility

#-------------------------------------
#   Initialize plots
#-------------------------------------
subgrad_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SGD for e-net covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-4, 1e1)

subgrad_fun_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Function Error");
title(@sprintf("SGD for e-net covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)

mirror_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"Relative distance to solution set $\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("SMD for e-net covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)
ylim(1e-4, 1e1)

mirror_fun_plot = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Function Error");
title(@sprintf("SMD for e-net covariance estimation (r=%i, d=%i)", r,d));
xlim(0,maxIter)

#-----------------------------------------------------------------------------------------
#   Run method(s) for each dimension
#-----------------------------------------------------------------------------------------
for i=1:num_tests
    ηsubgrad = subgrad_stepsizes[i];
    ηmirror = mirror_stepsizes[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    pert = randn(d,r);
    Xinit = Xtrue + init_radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Run the two methods
    (subgrad_err_hist, subgrad_fun_hist) = solve_enet(Xinit, Xtrue, stoch_err,  maxIter,
                                                            fill(ηsubgrad, maxIter), method="subgradient")
    (mirror_err_hist, mirror_fun_hist) = solve_enet(Xinit, Xtrue, stoch_err,  maxIter,
                                                                fill(ηmirror, maxIter), method="mirror")

    # Add errors to plots
    plt[:figure](subgrad_plot[:number])
    semilogy(subgrad_err_hist, #color=step_colors[i],
                    label=@sprintf("step=%1.2e", ηsubgrad));

    plt[:figure](subgrad_fun_plot[:number])
    semilogy(subgrad_fun_hist, #color=step_colors[i],
                    label=@sprintf("step=%1.2e", ηsubgrad));

    plt[:figure](mirror_plot[:number])
    semilogy(mirror_err_hist, #color=step_colors[i],
                    label=@sprintf("step=%1.2e", ηmirror));

    plt[:figure](mirror_fun_plot[:number])
    semilogy(mirror_fun_hist, #color=step_colors[i],
                    label=@sprintf("step=%1.2e", ηmirror));
end

# Add legends to plots and save final plots
plt[:figure](subgrad_plot[:number])
legend(loc="upper right")
savefig("plots/sgd_enet_stepsize_test.pdf");

plt[:figure](subgrad_fun_plot[:number])
legend(loc="upper right")
savefig("plots/sgd_enet_stepsize_test_fun_vals.pdf");

plt[:figure](mirror_plot[:number])
legend(loc="upper right")
savefig("plots/smd_enet_stepsize_test.pdf");

plt[:figure](mirror_fun_plot[:number])
legend(loc="upper right")
savefig("plots/smd_enet_stepsize_test_fun_vals.pdf");
