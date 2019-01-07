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
                                      method::String="subgradient",
                                      clipped::Bool=false,
                                      verbose::Bool=true
                                    )
    # Basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    #   Initializations
    X = copy(X0);
    Xtest = copy(X0);
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

    # draw the stochastic a, b
    (A,B) = get_ab(XTtrue, stochErr, maxIter);
    true_empirical_value = compute_empirical_function(Xtrue, A, B);

    # Make vectors for keeping track of progress
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors
    fun_val = compute_empirical_function(X, A, B);
    fun_hist =  fill(NaN, maxIter);  # to keep track of function values (approximates)

    # for debugging
    # @printf("Initial distance to solution: %1.2e, ",
    #                         sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X)));
    # @printf("Initial function error: %1.2e \n", compute_empirical_function(X, A, B));

    #   Run the method
    for k=1:maxIter
        # get stochastic a, b
        a = A[:,k];
        b = B[k];
        XTa = X'*a;

        # update subgradient - in place
        subgrad!(G, XTa, a, b);
        thresh = tr(G' * X) - fun_val;

        if method=="subgradient"
            # update X - in place
            if clipped
                η = minimum(stepSizes[k], fun_val/sum(abs2,G));
            else
                η = stepSizes[k];
            end
            BLAS.axpy!(-η,G,X);
        elseif method=="mirror"
            # update V = ∇Φ(X) - η * G
            η = stepSizes[k];
            gradPhi = ( 2*c0 + 3*c1*norm(X,2) + 4*c2*sum(abs2, X) ) * X;
            V = gradPhi  - η*G;
            nrmV = norm(V,2);

            # root finding problem to find λ = norm of X
            #           (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  = nrmV
            # (note that direction of X is identical to direction of V)
            λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2]);

            # update X - in place
            lmul!(0.0, Xtest)  # reset to zero
            BLAS.axpy!(λ/nrmV,V,Xtest)

            # check that X satisfied clipped constraints
            if clipped & (tr(G' * Xtest) < thresh)
                # first try: ∇ϕ(Y) = ∇ϕ(X_k)
                V = gradPhi;
                nrmV = norm(V,2);
                λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2]);
                lmul!(0.0, Xtest)  # reset to zero
                BLAS.axpy!(λ/nrmV,V,Xtest)

                # second try
                if (tr(G' * Xtest) > thresh)
                    alpha = tr(G' * gradPhi);
                    W = gradPhi - alpha*G;
                    a = 2*c0;
                    b = 3*c1;
                    c = 4*c2;
                    sqnormG = sum(abs2,G);
                    d = thresh^2/sqnormG;
                    # have (a + bλ + cλ^2)^2 (λ^2 - d) - ||W||^2 = 0
                    lam = get_root([ -a^2*d - sum(abs2,W),
                                            -2*a*b*d,
                                            a^2 - 2*a*c*d - b^2*d,
                                            2*a*b - 2*b*c*d,
                                            2*a*c + b^2 - c^2*d,
                                            2*b*c,
                                            c^2
                                            ]);
                    mu = -(2*c0+3*c1*lam+4*c2*lam^2)*thresh/sqnormG+alpha;
                    if mu >= 0
                        V = W + (alpha-mu)*G;
                        Xtest = (lam/nrmV) * V;
                    else
                        Xtest = X;  # don't take a step at all
                    end
                end
            end

            X = Xtest;

        end

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;

        fun_val = compute_empirical_function(X, A, B);
        value_error = fun_val - true_empirical_value ;
        fun_hist[k] = value_error;

        if verbose
            @printf("iter %3d: function err = %1.2e, dist err = %1.2e, stepsize = %1.2e\n",
                            k, value_error, normalized_err, η);
        end
    end

    return (err_hist, fun_hist)
end
