#----------------------------------------------------------------------------------------------------
# Solve covariance estimation problem using subgradient method
#----------------------------------------------------------------------------------------------------
include("func.jl");
using Printf

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

    # for keeping track of progress
    err = NaN;
    err_hist =  fill(NaN, maxIter);  # to keep track of errors
    fun_val = NaN;
    fun_hist =  fill(NaN, maxIter);  # to keep track of function values (approximates)

    # draw the stochastic a, b
    (A,B) = get_ab(XTtrue, stdev_stoch, maxIter)

    #   Run subgradient method
    for k=1:maxIter
        # get stochastic a, b
        a = A[:,k];
        b = B[k];
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

        val = compute_empirical_fun_val(X, A, B);
        fun_hist[k] = val;

        @printf("iter %3d: emp val = %1.2e, error = %1.2e, stepsize = %1.2e\n", k, val, normalized_err, η);
    end

    return (err_hist, fun_hist)
end
