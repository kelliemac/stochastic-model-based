#----------------------------------------------------------------------------------------------------
# Solve elastic net covariance estimation problem using mirror descent
#----------------------------------------------------------------------------------------------------
include("func.jl");
using Printf

function solve_enet_mirror(X0, Xtrue, steps_vec, maxIter, stdev_stoch)
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    # Coefficients for the Bregman divergence polynomials p and Φ
    #       p(u) = a0 + a1 * u + a2 * u² + a3 * u³
    #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴ + c3 ||x||₂⁵
    (a0, a1, a2, a3) = (1.0, 0.0, 1.0, 1.0);
    (c0, c1, c2, c3) = (a0*7/2,  a1*10/3,  a2*13/454, a3*16/5);

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

    # for keeping track of progress
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors
    fun_val = NaN;
    fun_hist =  fill(NaN, maxIter);  # to keep track of function values (approximates)

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
        sqnrmX = sum(abs2, X);
        V = ( 2*c0 + 3*c1*sqrt(sqnrmX) + 4*c2*sqnrmX + 5*c3*sqnrmX^(3/2) ) * X  - η*G;
        nrmV = norm(V,2);

        # root finding problem to find λ = norm of X
        #           (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  + (5*c3) * λ^4 = nrmV
        # (note that direction of X is identical to direction of V)
        λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2, 5*c3]);

        # update X - in place
        lmul!(0.0, X)  # reset to zero
        BLAS.axpy!(λ/nrmV,V,X)

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;

        fun_val = compute_empirical_fun_val(X, A, B);
        fun_hist[k] = fun_val;

        @printf("iter %3d: emp val = %1.2e, error = %1.2e, stepsize = %1.2e\n", k, fun_val, normalized_err, η);
    end

    return (err_hist, fun_hist)
end
