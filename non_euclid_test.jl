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
using LaTeXStrings
# pyplot(size = (800,600), legend = false)    # defaults for Plots
include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 10;
dims = [10];
r = 2;  # rank
σ = 0.001; # gaussian noise level in observations b

clf();
xlabel(L"Iteration $k$");
ylabel(L"$\| X - X_{true} \|_2^2$");

srand(123);

for d in dims
    #-----------------
    #   Data
    #-----------------
    X_true = randn(d,r);
    # noise = randn(d,d);
    # M = X_true * X_true.' + σ * noise;
    # nrmM = norm(M,1);
    stepSizes = fill(0.0001, maxIter);

    #-----------------
    #   Initialize
    #-----------------
    radius = 0.01;
    pert = radius * randn(d,r);
    X_init = X_true + pert;

    # for updating in place and keeping track of error over time
    X = copy(X_init);
    aTX = fill(0, (1,r))
    res = 0;
    G =  fill(0, (d,r));
    err_hist =  fill(NaN, maxIter);     # track function values

    #--------------------------------------------
    #   Run subgradient method
    #--------------------------------------------
    for k=1:maxIter
        # pick stochastic a, b
        a = randn(d,1)
        atransposeX!(aTX, X, a);

        getb!(b, aTX, σ);
        residuals!(res, aTX, b);
        subgrad!(G, res, a, aTX);

         # must be smaller than 1 / (weak convexity constant); oscillates with constant step
        BLAS.axpy!(-stepSizes[i],G,X);

        # record status and print to console
        error = sum(abs2, X-Xtrue)
        err_hist[k] = error;
        @printf("iteration %3d, error = %1.2e, stepsize = %1.2e\n", k, error, stepSizes[i]);
    end

    #------------------------
    #   Plot results
    #------------------------
    semilogy(err_hist);
end

savefig("non_eucl_test.pdf");
