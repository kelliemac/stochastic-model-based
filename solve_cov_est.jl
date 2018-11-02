#-------------------------------------------------------------------------------------------------------------
#   Solve covariance estimation problem
#           min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   using either 1) subgradient method or
#                       2) mirror descent (polynomial mirror map)
#
#   Inputs:     X0 = initialization (d x r matrix)
#                   Xtrue = true matrix we are solving for (so we can plot dist. to solution)
#                   stepSizes = one-dim array of stepsizes (length maxIter)
#                   maxIter = maximum number of iterations
#                   stochErr = variance of stochastic errors in residuals <XX^T,aa^T> - b
#                   method = "subgradient" or "mirror"
#
#   Outputs:   (err_hist, fun_hist) = (history of normalized distances to solution,
#                                                       history of empirical function values)
#-------------------------------------------------------------------------------------------------------------
include("func.jl");
using Printf

function solve_cov_est(X0::Array{Float64,2},
                                      Xtrue::Array{Float64,2},
                                      stochErr::Float64,
                                      maxIter::Int64,
                                      stepSizes::Array{Float64,1};
                                      method::String="subgradient"
                                    )
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    #   Initializations
    X = copy(X0);
    a = zeros(d);
    XTa = zeros(r);
    b = 0;
    res = 0;
    G = zeros(d,r);
    η = 0;

    if method=="mirror"
        # Coefficients for the Bregman divergence polynomials p and Φ
        #       p(u) = a0 + a1 * u + a2 * u²
        #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴
        (a0, a1, a2) = (1.0, 0.0, 1.0);
        (c0, c1, c2) = (a0*7/2,  a1*10/3,  a2*13/4);

        #   more initializations
        V = zeros(d,r);
        λ = 0;
    end

    # Make vectors for keeping track of progress
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors
    fun_val = NaN;
    fun_hist =  fill(NaN, maxIter);  # to keep track of function values (approximates)

    # draw the stochastic a, b
    (A,B) = get_ab(XTtrue, stochErr, maxIter)

    #   Run the method
    for k=1:maxIter
        # get stochastic a, b
        a = A[:,k];
        b = B[k];
        XTa = X'*a;

        # update subgradient - in place
        subgrad!(G, XTa, a, b);

        if method=="subgradient"
            # update X - in place
            η = stepSizes[k];
            BLAS.axpy!(-η,G,X);
        elseif method=="mirror"
            # update V = ∇Φ(X) - η * G
            η = stepSizes[k];
            V = ( 2*c0 + 3*c1*norm(X,2) + 4*c2*sum(abs2, X) ) * X  - η*G;
            nrmV = norm(V,2);

            # root finding problem to find λ = norm of X
            #           (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  = nrmV
            # (note that direction of X is identical to direction of V)
            λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2]);

            # update X - in place
            lmul!(0.0, X)  # reset to zero
            BLAS.axpy!(λ/nrmV,V,X)
        end

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
