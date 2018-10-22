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
#       res = <XX^T, a a^T> - b = || aTX ||₂² - b  (residual)
#------------------------------------------------------------------------------------------------------------

using LinearAlgebra
using Printf
using Roots

#----------------------------------------------------------------------------
# Function to draw stochastic a and b
#----------------------------------------------------------------------------
function get_ab(d, XTtrue, stdev_stoch)
    a = randn(d);
    stoch_err = stdev_stoch * randn();  # normal(0,stdev_stoch²) errors
    b = sum(abs2, XTtrue*a) + stoch_err;
    return(a,b)
end

#----------------------------------------------------------------------------
# Function to update the subgradient G, in place
#  Note G = 2 * sign(res) * (XTa)ᵀ
#----------------------------------------------------------------------------
function subgrad!(G, XTa, a, b)
    res = sum(abs2, XTa) - b;
    lmul!(0.0,G)  # reset G to zero
    BLAS.ger!(2*sign.(res), a, XTa, G);  # rank one update
end

#----------------------------------------------------------------------------
# Solve using subgradient method
#----------------------------------------------------------------------------
function solve_cov_est_subgradient(X0, Xtrue, steps_vec, maxIter, stdev_stoch)
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
    val = NaN;
    vals =  fill(NaN, maxIter);  # to keep track of function values (approximates)

    #   Run subgradient method
    for k=1:maxIter
        # draw stochastic a, b
        (a,b) = get_ab(d, XTtrue, stdev_stoch);
        XTa = X'*a;

        # update subgradient - in place
        subgrad!(G, XTa, a, b);

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
function solve_cov_est_mirror(X0, Xtrue, steps_vec, maxIter, stdev_stoch)
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    # Coefficients for the Bregman divergence polynomials p and Φ
    #       p(u) = a0 + a1 * u + a2 * u²
    #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴
    (a0, a1, a2) = (1, 0, 1);
    (c0, c1, c2) = (a0*7/2,  a1*10/3,  a2*13/4);

    #   Initialize
    X = copy(X0);
    a = zeros(d);
    XTa = zeros(r);
    b = 0;
    res = 0;
    G = zeros(d,r);
    V = zeros(d,r);
    η = 0;
    λ = 0;
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors

    #   Run mirror descent method, i.e. at each iteration X_{k+1} solves
    #               ∇ Φ(X_{k+1})  =   ∇ Φ (X_k) - η_k * G_k
    #  where G_k is the stochastic subgradient at X_k.
    #  Note that here,
    #               ∇ Φ(X) = (2*c0 +3*c1*||X||₂ +4*c2*||X||₂²) * X
    for k=1:maxIter
        # draw stochastic a, b
        (a,b) = get_ab(d, XTtrue, stdev_stoch);
        XTa = X'*a;

        # update subgradient - in place
        subgrad!(G, XTa, a, b);

        # update V = ∇Φ(X) - η * G
        η = steps_vec[k];
        V = ( 2*c0 + 3*c1*norm(X,2) + 4*c2*sum(abs2, X) ) * X  - η*G;
        nrmV = norm(V,2);

        # root finding problem to find λ = norm of X
        #           (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  = α
        # (note that direction of X is identical to direction of V)
        try
            λ = find_zero(λ -> (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3 - nrmV, (0,Inf), nrmV);
        catch y
            if isa(y, Roots.ConvergenceFailed)
                print("root finding failed, ending iteration\n")
                return(err_hist)
            end
        end

        # update X - in place
        lmul!(0.0, X)  # reset to zero
        BLAS.axpy!(λ,V,X)

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;
        @printf("iteration %3d: error = %1.2e, stepsize = %1.2e\n", k, normalized_err, η);
    end

    return err_hist
end
