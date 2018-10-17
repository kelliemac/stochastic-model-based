#------------------------------------------------------------------------------------------------------------
#   Define functions needed to solve
#   min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   via
#   1) stochastic subgradient method (Φ = 1/2 ||⋅||^2)
#   2) stochastic mirror descent (Φ = a quartic polynomial)
#
#   Stochastic subgradients are
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X
#   where (a, b) ∈ Rᵐ x R are chosen randomly.
#
#   We use the notation
#       aTX = a^T X
#       R = <XX^T, a a^T> - b = norm(aTX , 2) - b  (residual)
#------------------------------------------------------------------------------------------------------------

using LinearAlgebra
using Printf
using Roots

#----------------------------------------------------------------------------
# Solve using subgradient method
#----------------------------------------------------------------------------
function solve_cov_est_subgradient(X0, Xtrue, steps_vec, maxIter)
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    #   Initialize
    X = copy(X0);
    a = zeros(d);
    XTa = zeros(r);
    b = 0;
    res = 0;
    G = zeros(d,r);
    η = 0;
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors

    #   Run subgradient method
    for k=1:maxIter
        # draw stochastic a, b; update resulting residual
        a = randn(d);
        XTa = X'*a;
        res = randn();
        res = 0.001 * res / norm(res,2)
        b = sum(abs2, XTa) - res;

        # update subgradient - in place
        # note G = 2 * sign(res) * (XTa)ᵀ
        lmul!(0.0,G)  # reset G to zero
        BLAS.ger!(2*sign.(res), a, XTa, G);  # rank one update

        # update X - in place
        η = steps_vec[k];
        BLAS.axpy!(-η,G,X);

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;
        @printf("iteration %3d: error = %1.2e, stepsize = %1.2e\n", k, normalized_err, η);
    end

    return err_hist
end

#----------------------------------------------------------------------------
# Solve using mirror descent
#----------------------------------------------------------------------------
function solve_cov_est_mirror(X0, Xtrue, steps_vec, maxIter)
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    # Coefficients for the Bregman divergence polynomials p and Φ
    #       p(u) = a0 + a1 * u + a2 * u²
    #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴
    a0 = 1;
    a1 = 0;
    a2 = 1;

    c0 = a0*7/2;
    c1 = a1*10/3;
    c2 = a2*13/4;

    #   Initialize
    X = copy(X0);
    a = zeros(d);
    XTa = zeros(r);
    b = 0;
    res = 0;
    G = zeros(d,r);
    η = 0;
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors

    #   Run mirror descent method, i.e. at each iteration solve
    #  x_{k+1}  = arg min ₓ  {    }
    #
    for k=1:maxIter
        # draw stochastic a, b; update resulting residual
        a = randn(d);
        XTa = X'*a;
        res = randn();
        res = 0.001 * res / norm(res,2)
        b = sum(abs2, XTa) - res;

        # update subgradient - in place
        # note G = 2 * sign(res) * (XTa)ᵀ
        lmul!(0.0,G)  # reset G to zero
        BLAS.ger!(2*sign.(res), a, XTa, G);  # rank one update

        # update direction of X: same as ∇Φ(X) - η * G
        V =   ( 2*c0 + 4*c2*sum(abs2, X) ) * X  - η*G   # ∇Φ(X) - η * G
        α = norm(V,2)

        # root finding problem to find λ = norm of X
        λ = try
            find_zero(λ -> (2*c0) * λ + (4*c2) * λ^3 - α, 1.0)
        catch y
            if isa(y, Roots.ConvergenceFailed)
                print("root finding failed, ending iteration\n")
                break
            end
        end

        # rescale X
        lmul!( λ/α, X )

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;
        @printf("iteration %3d: error = %1.2e, stepsize = %1.2e\n", k, normalized_err, η);
    end

    return err_hist
end
