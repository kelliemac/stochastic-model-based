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
#------------------------------------------------------------------------------------------------------------
workspace()

using PyPlot
# using LaTeXStrings
# pyplot(size = (800,600), legend = false)    # defaults for Plots
include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 1000;
dims = [100, 300, 500];
σ = 0.001;  # noise level in observation

clf();
xlabel(L"Iteration $k$");
ylabel(L"$\| \theta_k\theta_k^T - M \|_1 / \|M\|_1$");

srand(123);

for n in dims
    #-----------------
    #   Data
    #-----------------
    X_true = randn(n);
    noise = randn(n,n);
    M = X_true * X_true.' + σ * noise;
    nrmM = norm(M,1);

    #-----------------
    #   Initialize
    #-----------------
    radius = 1.0;
    pert = randn(n);
    X_init = X_true + (radius * norm(X_true,1) / norm(pert,1) ) * pert;

    X = copy(X_init);
    err_hist =  fill(NaN, maxIter);     # track function values

    R = X*X' - M;
    S = map(sign,R);
    g = 2 * S' * θ;

    #--------------------------------------------
    #   Run subgradient method
    #--------------------------------------------
    for k=1:maxIter
        residuals!(R, θ, M);
        subgrad!(g, θ, M, R, S);

        stepSize = 0.0001;  # must be smaller than 1 / (weak convexity constant); oscillates with constant step
        BLAS.axpy!(-stepSize,g,θ);

        # record status and print to console
        obj = objective(R);
        err_hist[k] = obj / nrmM;
        @printf("iter %3d, obj %1.2e, step %1.2e\n", k, obj, stepSize);
    end

    #------------------------
    #   Plot results
    #------------------------
    semilogy(err_hist);
end

savefig("non_eucl_test.pdf");
