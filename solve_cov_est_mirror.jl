#----------------------------------------------------------------------------------------
# Solve covariance estimation problem using mirror descent
#----------------------------------------------------------------------------------------
include("func.jl");
using Printf

function solve_cov_est_mirror(X0, Xtrue, steps_vec, maxIter, stdev_stoch)
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    # Coefficients for the Bregman divergence polynomials p and Φ
    #       p(u) = a0 + a1 * u + a2 * u²
    #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴
    (a0, a1, a2) = (1.0, 0.0, 1.0);
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

    # draw the stochastic a, b
    (A,B) = get_ab(XTtrue, stdev_stoch, maxIter)

    #   Run mirror descent method, i.e. at each iteration X_{k+1} solves
    #               ∇ Φ(X_{k+1})  =   ∇ Φ (X_k) - η_k * G_k
    #  where G_k is the stochastic subgradient at X_k.
    #  Note that here,
    #               ∇ Φ(X) = (2*c0 +3*c1*||X||₂ +4*c2*||X||₂²) * X
    for k=1:maxIter
        # get stochastic a, b
        a = A[:,k];
        b = B[k];
        XTa = X'*a;

        # update subgradient - in place
        subgrad!(G, XTa, a, b);

        # update V = ∇Φ(X) - η * G
        η = steps_vec[k];
        V = ( 2*c0 + 3*c1*norm(X,2) + 4*c2*sum(abs2, X) ) * X  - η*G;
        nrmV = norm(V,2);

        # root finding problem to find λ = norm of X
        #           (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  = nrmV
        # (note that direction of X is identical to direction of V)
        λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2]);

        # update X - in place
        lmul!(0.0, X)  # reset to zero
        BLAS.axpy!(λ/nrmV,V,X)

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;
        @printf("iter %3d: nrmX = %1.2e, nrmG = %1.2e, error = %1.2e, stepsize = %1.2e\n", k, norm(X,2), norm(G,2), normalized_err, η);
    end

    return err_hist
end
