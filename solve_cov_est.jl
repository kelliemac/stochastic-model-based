# ---------------------------------------------------------------------------------------------
#   Solve covariance estimation problem
#           min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   using either 1) subgradient method or
#                       2) mirror descent (with a polynomial mirror map)
#
#   Inputs:     X0 = initialization (d x r matrix)
#                   Xtrue = true matrix (used to plot dist. to solution)
#                   stochErr = variance of stochastic errors in residuals <XX^T,aa^T> - b
#                   maxIter = maximum no. iterations
#                   stepSizes = one-dim array of stepsizes (length maxIter)
#                   method = "subgradient" or "mirror"
#                   clipped = whether to use clipping (true/false)
#                   verbose = whether to print output (true/false)
#
#   Outputs:   (err_hist, fun_hist) = (array of normalized distances to solution,
#                                               array of errors in empirical function values)
# ---------------------------------------------------------------------------------------------
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
    # Extract basic data
    (d,r) = size(X0);
    XTtrue = Xtrue';
    sqnrmXtrue = sum(abs2, Xtrue);

    #  Initialize
    X = copy(X0);
    a = zeros(d);
    XTa = zeros(r);
    b = 0;
    res = 0;  # residual <XX^T, aa^T> - b
    G = zeros(d,r);  # subgradient at X
    η = 0;  # stepsize

    if method=="mirror"
        # Set coefficients for the Bregman divergence polynomials
        #       p(u) = a0 + a1 * u + a2 * u²
        #       Φ(x) = c0 ||x||₂² + c1 ||x||₂³ + c2 ||x||₂⁴   (mirror map)
        (a0, a1, a2) = (1.0, 0.0, 1.0);  # chosen by us
        (c0, c1, c2) = (a0*7/2,  a1*10/3,  a2*13/4);  # set according to theorem

        # Initialize more quantities
        Y = copy(X0);  # need an extra copy of current iterate
        V = zeros(d,r);  # V = ∇Φ(X) - η * G
        λ = 0;  # λ = ||X||
    end

    # Draw stochastic a, b for all iterations at once
    (A,B) = get_ab(XTtrue, stochErr, maxIter);
    true_empirical_value = compute_empirical_function(Xtrue, A, B);

    # Initialize vectors for tracking progress
    err = NaN;  # distance to solution
    err_hist =  fill(NaN, maxIter);  # distances to solution
    fun_val = compute_empirical_function(X, A, B);  # empirical function value
    fun_hist =  fill(NaN, maxIter);  # empirical function values

    #   Run the method (stochatic subgradient or stochastic mirror)
    for k=1:maxIter
        # Get stochastic a, b
        a = A[:,k];
        b = B[k];
        XTa = X'*a;

        # Update subgradient in place
        subgrad!(G, XTa, a, b);
        thresh = dot(G, X) - fun_val;  # clipping threshold

        # Update X in place
        if method=="subgradient"
            # Get step size
            if clipped
                η = min(stepSizes[k], fun_val/sum(abs2,G));
            else
                η = stepSizes[k];
            end
            BLAS.axpy!(-η,G,X);
        elseif method=="mirror"
            # Update V = ∇Φ(X) - η * G
            η = stepSizes[k];
            gradPhi = ( 2*c0 + 3*c1*norm(X,2) + 4*c2*sum(abs2, X) ) * X;
            V = gradPhi  - η*G;
            nrmV = norm(V,2);

            # Note that direction of X is identical to direction of V.
            # Root finding problem to find λ = ||X|| is
            #    (2*c0) * λ + (3*c1) * λ^2 + (4*c2) * λ^3  = nrmV
            λ = get_root([-nrmV, 2*c0, 3*c1, 4*c2]);

            # Record intermediate matrix Y
            lmul!(0.0, Y)  # reset to zero
            BLAS.axpy!(λ/nrmV, V, Y)

            if (! clipped) | (dot(G,Y) >= thresh)
                X = Y;
            else  # Need to find a better Y.
                # First try: ∇ϕ(Y) = ∇ϕ(X_k), i.e. Lagrange multiplier is zero
                nrmGrad= norm(gradPhi,2);
                λ = get_root([-nrmGrad, 2*c0, 3*c1, 4*c2]);
                lmul!(0.0, Y)  # reset to zero
                BLAS.axpy!(λ/nrmGrad, gradPhi, Y)

                if (dot(G,Y) <= thresh)
                    X = Y;
                else  # Y still doesn't satisfy clipped constraint
                    # Second try: use gradPhi = W + α*G,  W ⟂ G,  and ∇Φ(Y) = W + (α-μ)*G
                    sqnormG = sum(abs2,G);
                    α = dot(G, gradPhi) / sqnormG;
                    W = gradPhi - α*G;

                    #  Inner product with G + norm squared gives coefficients
                    d0, d1, d2 = 2*c0, 3*c1, 4*c2;
                    d3 = thresh^2/sqnormG;

                    # Now solve for λ=||Y|| using (a + bλ + cλ^2)^2 (λ^2 - d) - ||W||^2 = 0
                    λ = get_root([ -d0^2*d3 - sum(abs2,W),
                                        -(2*d0*d1)*d3,
                                        d0^2 - (2*d0*d2-d1^2)*d3,
                                        (2*d0*d1) - (2*d1*d2)*d3,
                                        (2*d0*d2 + d1^2) - d2^2*d3,
                                        2*d1*d2,
                                        d2^2
                                        ]);

                    # Get μ from λ
                    μ = α  - (d0+d1*λ+d2*λ^2) * thresh / sqnormG ;
                    if μ > 0
                        V = W + (α-μ)*G;
                        Y = (λ/nrmV) * V;
                        X = Y;
                    else
                        # Can't find a good Y, so don't take a step at all. Keep X as is.
                    end
                end
            end
        end

        # record error status and print to console
        err = sqnrmXtrue + sum(abs2, X) - 2 * sum(svdvals(XTtrue * X));
        normalized_err = err / sqnrmXtrue;
        err_hist[k] = normalized_err;

        fun_val = compute_empirical_function(X, A, B);
        # value_error = fun_val - true_empirical_value;
        fun_hist[k] = fun_val;

        if verbose
            @printf("iter %3d: function err = %1.2e, dist err = %1.2e, stepsize = %1.2e\n",
                            k, value_error, normalized_err, η);
        end
    end

    return (err_hist, fun_hist)
end
