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
#       R = <XX^T, a a^T> - b = norm(aTX , 2) - b  (residual)
#       S = sign(R)
#------------------------------------------------------------------------------------------------------------

function atransposeX!(aTX, X, a)
    # gemv!('T', 1.0, a, X, 0.0, aTX);
    aTX = a'*X;
end

# Update one-dim residual in place
function residuals!(res, aTX, b)
    res = sum(abs2, aTX)  - b;
end

# generate b as true measurement plus random error
function getb!(b, aTX, σ)
    b = sum(abs2, aTX)  + σ*randn(1);
end

# Compute one-dim residual and return value
function residuals!(res, aTX, b)
    res = sum(abs2, aTX)  - b;
end

# Update the subgradient
#       G(X) = S * 2 aa^T X
function subgrad!(G, res, a, aTX)
    BLAS.gemv!('N', 2*sign(res), a, aTX, 0.0, G);
end
