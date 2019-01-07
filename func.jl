#------------------------------------------------------------------------------------------------------------
#   Define functions needed to solve the covariance estimation problem:
#           min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   and it's elastic net variant with
#           F(X) :=  E_{a,b} | <XX^T, aa^T> - b | + (<XX^T, aa^T> - b )^2
#
#   Solve via two methods:
#           1) stochastic subgradient (Φ = 1/2 ||⋅||^2)
#           2) stochastic mirror descent (Φ = a polynomial)
#
#   Note that stochastic subgradients are
#           ∂ (  | <XX^T, a a^T> - b |  )   =    sign(res) * 2 aa^T X
#           ∂ (  | <XX^T, a a^T> - b |  )   =    (2 * res + sign(res) ) * 2 aa^T X
#   where (a, b) ∈ Rᵐ x R are chosen randomly, and
#           res = <XX^T, a a^T> - b = || aTX ||₂² - b  (residual)
#------------------------------------------------------------------------------------------------------------

using LinearAlgebra
using Polynomials
using Random

#------------------------------------------------------------------------------------------
# Draw stochastic a's (gaussian) and corresponding b's
# Returns: maxIter of them, in format:
#                       A = matrix with a's in columns
#                       B = vector with b's in entries
#------------------------------------------------------------------------------------------
function get_ab(XTtrue, stochErr::Float64, maxIter::Int64; noiseType::String="gaussian")
    (r,d) = size(XTtrue);
    A = randn(d, maxIter);  # gaussian entries in a's
    bError = stochErr * randn(1, maxIter);  # normal(0,stochErr²) errors
    if noiseType=="sparse"
        p = 0.25; # probability of corrupting a measurement
        noiseless_idxs = randsubseq(collect(1:maxIter), 1-p);
        bError[noiseless_idxs] = zeros(length(noiseless_idxs));
    end
    B = sum(abs2, XTtrue*A, dims=1) + bError;
    return(A,B)
end

#---------------------------------------------------------------------------------------------------
#   Compute empirical function values, based on (A,B) as above
#   Inputs:     X = point to evaluate function (d x r matrix)
#                   A = matrix with columns as the a's
#                   B = vector with b's as entries
#---------------------------------------------------------------------------------------------------
# vanilla covarianc estimation function value
function compute_empirical_function(X, A, B)
    residuals = sum(abs2, X'*A, dims=1) - B;
    return sum(abs, residuals) / size(A,2)
end

# elastic net empirical function value
function compute_enet_empirical_function(X, A, B)
    residuals = sum(abs2, X'*A, dims=1) - B;
    return (sum(abs, residuals) + sum(abs2, residuals)) / size(A,2)
end

#----------------------------------------------------------------------------
# Update the subgradient G, in place
#----------------------------------------------------------------------------
function subgrad!(G, XTa, a, b)
    res = sum(abs2, XTa) - b;
    lmul!(0,G)  # reset G to zero
    BLAS.ger!(2*sign(res), a, XTa, G);  # rank one update
end

function enet_subgrad!(G, XTa, a, b)
    res = sum(abs2, XTa) - b;
    lmul!(0.0,G)  # reset G to zero
    BLAS.ger!(2*sign.(res)+2*res, a, XTa, G);  # rank one update
end

#------------------------------------------------------------------------------------------------------------
# Find the smallest real, positive root of a polynomial
# input : coefficients, as a vector by increasing degree (e.g. [a0, a1, a2, a3])
#------------------------------------------------------------------------------------------------------------
function get_root(coeffs)
    all_roots = roots(Poly(coeffs));

    # reduce to positive real roots
    pos_real_idxs = map(x->(imag(x)==0.0 && real(x)>0), all_roots);
    pos_real_roots = real(all_roots[pos_real_idxs]);

    # return smallest positive real root
    return(minimum(pos_real_roots))
end
