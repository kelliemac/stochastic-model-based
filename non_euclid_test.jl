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
using PyPlot
using Random
using LaTeXStrings
using LinearAlgebra.BLAS
using Printf
# pyplot(size = (800,600), legend = false)    # defaults for Plots
include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 1000;
dims = [10, 100];
r = 2;  # rank
σ = 0.001; # gaussian noise level in observations b

clf();
xlabel(L"Iteration $k$");
ylabel(L"$\| X - X_{true} \|_2^2$");
title(@sprintf("Subgradient method for covariance estimation (r=%i)", r));

Random.seed!(123);

for d in dims
    #-----------------
    #   Data
    #-----------------
    X_true = randn(d,r);
    stepSizes = fill(0.0001, maxIter);

    #-----------------
    #   Initialize
    #-----------------
    radius = 0.1;
    pert = radius * randn(d,r);
    X_init = X_true + pert;

    X = copy(X_init);
    G = zeros(d,r);
    err_hist =  fill(NaN, maxIter);     # track function values

    #--------------------------------------------
    #   Run subgradient method
    #--------------------------------------------
    for k=1:maxIter
        # pick stochastic a, b
        a = randn(d);
        aTX = a'*X;
        XTa = X'*a;
        b = sum(abs2, XTa)  + σ*randn();

        res = sum(abs2, XTa)  - b;
        BLAS.ger!(2*sign.(res), a, XTa, G);

         # must be smaller than 1 / (weak convexity constant); oscillates with constant step
        BLAS.axpy!(-stepSizes[k],G,X);

        # record status and print to console
        this_error = sum(abs2, X-X_true);
        err_hist[k] = this_error;
        @printf("iteration %3d, error = %1.2e, stepsize = %1.2e\n", k, this_error, stepSizes[k]);
    end

    #------------------------
    #   Plot results
    #------------------------
    semilogy(err_hist);
end

savefig("non_eucl_test.pdf");
