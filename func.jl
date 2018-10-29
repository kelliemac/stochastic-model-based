#------------------------------------------------------------------------------------------------------------
#   Define functions needed to solve
#   min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   and elastic net version:
#   F(X) :=  E_{a,b} | <XX^T, aa^T> - b | + (<XX^T, aa^T> - b )^2
#
#
#
#   via
#   1) stochastic subgradient method (Φ = 1/2 ||⋅||^2)
#   2) stochastic mirror descent (Φ = a quartic polynomial)
#
#   Stochastic subgradients are
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X +
#   where (a, b) ∈ Rᵐ x R are chosen randomly.
#
#   We use the notation
#       aTX = a^T X
#       res = <XX^T, a a^T> - b = || aTX ||₂² - b  (residual)
#------------------------------------------------------------------------------------------------------------

using LinearAlgebra
using Polynomials

#------------------------------------------------------------------------------------------
# Function to draw maxIter stochastic a and b
# Returns: matrix whose columns are the a's, and
#                vector whose entries are the corresponding b's.
#------------------------------------------------------------------------------------------
function get_ab(XTtrue, stdev_stoch, maxIter)
    (r,d) = size(XTtrue);

    # generate A
    A = randn(d, maxIter);

    # generate noisy b based on A
    stoch_err = stdev_stoch * randn(1, maxIter);  # normal(0,stdev_stoch²) errors
    XTA = XTtrue*A;
    B = sum(abs2, XTA, dims=1) + stoch_err;

    return(A,B)
end

#---------------------------------------------------------------------------------------------------
# Function to compute empirical function value,
# using stochastic a's and b's
# Input: matrixA with columns as the a's, vector B with entries as the b's
#---------------------------------------------------------------------------------------------------
function compute_empirical_fun_val(X, A, B)
    residuals = sum(abs2, X'*A, dims=1) - B;
    return sum(abs, residuals)
end

function compute_enet_empirical_fun_val(X, A, B)
    residuals = sum(abs2, X'*A, dims=1) - B;
    return sum(abs, residuals) + sum(abs2, residuals)
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

function enet_subgrad!(G, XTa, a, b)
    res = sum(abs2, XTa) - b;
    lmul!(0.0,G)  # reset G to zero
    BLAS.ger!(2*sign.(res)+2*res, a, XTa, G);  # rank one update
end

#------------------------------------------------------------------------------------------------------------
# Function to find the smallest real, positive root of a polynomial
# input : coefficients, as a vector by increasing degree (e.g. [a0, a1, a2, a3])
#------------------------------------------------------------------------------------------------------------
function get_root(coeffs)
    all_roots = roots(Poly(coeffs));

    # reduce to positive real roots
    pos_real_idxs = map(x->(imag(x)==0.0 && real(x)>0), all_roots);
    pos_real_roots = real(all_roots[pos_real_idxs]);

    # return smallest real root
    return(minimum(pos_real_roots))
end
