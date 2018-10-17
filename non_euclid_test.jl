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
using Random
using LaTeXStrings
using LinearAlgebra
using Printf
using PyPlot

include("func.jl");

#-------------------------------------
#   Parameters
#-------------------------------------
maxIter = 100;
dims = [10, 100];
r = 2;  # rank
σ = 0.001; # gaussian noise level in observations b

clf();
figure(figsize=[10,6]);
xlabel(L"Iteration $k$");
ylabel(L"$\frac{\| X_k - X_{true} \|_2^2 }{\| X_{true} \|_2^2}$");
title(@sprintf("Subgradient method for covariance estimation (r=%i)", r));
xlim(0,maxIter)

Random.seed!(123);

for d in dims
    #-----------------
    #   Data
    #-----------------
    Xtrue = randn(d,r);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);
    stepSizes = fill(0.001, maxIter);

    #-----------------
    #   Initialize
    #-----------------
    radius = 0.1;
    pert = radius * randn(d,r);
    X_init = Xtrue + pert;

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

        # update residual
        res = sum(abs2, XTa)  - b;

        # update subgradient - in place
        # note G = 2 * sign(res) * (XTa)ᵀ
        lmul!(0.0,G)  # reset G to zero
        BLAS.ger!(2*sign.(res), a, XTa, G);  # rank one update

        # update X - in place
        η = stepSizes[k];
        BLAS.axpy!(-η,G,X);

        # record status and print to console
        this_error = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        err_hist[k] = this_error;
        @printf("iteration %3d, error = %1.2e, stepsize = %1.2e\n", k, this_error, η);
    end

    #------------------------
    #   Plot results
    #------------------------
    semilogy(err_hist, label=@sprintf("d=%i", d));
end

legend(loc="lower right")
savefig("non_eucl_test.pdf");
