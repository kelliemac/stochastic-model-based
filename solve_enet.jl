#----------------------------------------------------------------------------------------------------
# Solve elastic net covariance estimation problem using subgradient method
#----------------------------------------------------------------------------------------------------
include("func.jl");
using Printf

function solve_enet(X0::Array{Float64,2},
                                Xtrue::Array{Float64,2},
                                stochErr::Float64,
                                maxIter::Int64,
                                stepSizes::Array{Float64,1};
                                method::String="subgradient",
                                verbose::Bool=true
                                )

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

    if method=="mirror"
        # Coefficients for the Bregman divergence polynomials p and Φ
        #       p(u) = a0 + a1 * u + a2 * u² + a3 * u³
        #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴ + c3 ||x||₂⁵
        (a0, a1, a2, a3) = (1.0, 0.0, 1.0, 1.0);
        (c0, c1, c2, c3) = (a0*7/2,  a1*10/3,  a2*13/454, a3*16/5);

        V = zeros(d,r);
        λ = 0;
    end

    # for keeping track of progress
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors
    fun_val = NaN;
    fun_hist =  fill(NaN, maxIter);  # to keep track of function values (approximates)

    # draw the stochastic a, b
    (A,B) = get_ab(XTtrue, stochErr, maxIter)
    true_empirical_value = compute_empirical_function(Xtrue, A, B);

    #   Run subgradient method
    for k=1:maxIter
        # get stochastic a, b
        a = A[:,k];
        b = B[k];
        XTa = X'*a;

        # update subgradient - in place
        enet_subgrad!(G, XTa, a, b);

        if method=="subgradient"
            # update X - in place
            η = stepSizes[k];
            BLAS.axpy!(-η,G,X);
        elseif method=="mirror"
            # update V = ∇Φ(X) - η * G
            η = stepSizes[k];
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
        end

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;

        fun_val = compute_enet_empirical_function(X, A, B);
        value_error = fun_val - true_empirical_value ;
        fun_hist[k] = value_error;

        if verbose
            @printf("iter %3d: emp val = %1.2e, error = %1.2e, stepsize = %1.2e\n",
                                k, value_error, normalized_err, η);
        end
    end

    return (err_hist, fun_hist)
end
