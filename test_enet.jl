#------------------------------------------------------------------------------------------------------------
#   Code to solve elastic net-ish version of covariance estimation
#   min_{X \in \R^{d x r}} F(X) :=  E_{a,b} | <XX^T, aa^T> - b | + (<XX^T, aa^T> - b )^2
#   for a bunch of trials
#------------------------------------------------------------------------------------------------------------
using Random  # for setting the random seed
using PyPlot
using LaTeXStrings, Printf  # for plot labels

include("solve_enet.jl");

#-------------------------------------
#   Set parameters
#-------------------------------------
r = 2;  # rank
dims = [100];  # ambient dimension
dims_colors = ["#1f78b4"]; #, "#33a02c"];
stoch_err = 0.1;  # standard deviation of errors in stochastic measurements b

maxIter = 2000;
dims_steps_subgrad = [1e-6];
dims_steps_mirror = [0.1];

Random.seed!(123);  # for reproducibility

#-------------------------------------
#   Initialize plot
#-------------------------------------
dist_fig = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"$\min_U \; \frac{\| X_k U - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("Distance to solution for elastic net covariance estimation (r=%i)", r));
xlim(0,maxIter)

funval_fig = figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel("Empirical Function Value");
title(@sprintf("Function values for elastic net covariance estimation (r=%i)", r));
xlim(0,maxIter)

#-----------------------------------------------------------------------------------------
#   Run stochastic subgradient method for each dimension
#-----------------------------------------------------------------------------------------
for i in 1:length(dims)
    d = dims[i];

    # Generate true matrix we are searching for
    Xtrue = randn(d,r);

    # Generate initial point
    radius = 1.0;  # 2-norm measure of relative deviation between Xtrue and Xinit
    pert = randn(d,r);
    Xinit = Xtrue + radius * (norm(Xtrue, 2) / norm(pert, 2)) * pert;

    # Specify step sizes
    steps_subgrad = fill(dims_steps_subgrad[i], maxIter);
    steps_mirror = fill(dims_steps_mirror[i], maxIter);

    # Run the two methods
    (err_hist_subgrad, fun_hist_subgrad) = solve_enet(Xinit, Xtrue, stoch_err, maxIter,
                                                                                        steps_subgrad,
                                                                                        method="subgradient")
    (err_hist_mirror, fun_hist_mirror) = solve_enet(Xinit, Xtrue, stoch_err, maxIter,
                                                                                        steps_mirror,
                                                                                        method="mirror")

    # Add distances to appropriate plot
    plt[:figure](dist_fig[:number])
    semilogy(err_hist_subgrad, linestyle="--", color=dims_colors[i],
                    label=@sprintf("SGD, d=%i", d));
    semilogy(err_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));

    # Add function values to appropriate plot
    plt[:figure](funval_fig[:number])
    semilogy(fun_hist_subgrad, linestyle="--", color=dims_colors[i],
                    label=@sprintf("SGD, d=%i", d));
    semilogy(fun_hist_mirror, color=dims_colors[i], label=@sprintf("SMD, d=%i", d));
end

# Add legends to plots and save final results
plt[:figure](dist_fig[:number])
legend(loc="lower right")
savefig("plots/enet_distances.pdf");

plt[:figure](funval_fig[:number])
legend(loc="lower right")
savefig("plots/enet_values.pdf");
