#------------------------------------------------------------------------------------------------------------
#   Define functions needed to solve
#   min_{X \in \R^{d x r}}   F(X)   :=    E_{a,b} | <XX^T, aa^T> - b |
#   via
#   1) stochastic subgradient method (Φ = 1/2 ||⋅||^2)
#   2) stochastic mirror descent (Φ = a quartic polynomial)
#
#   Stochastic subgradients are
#   ∂ (  | <XX^T, a a^T> - b |  )   =    sign(<XX^T, aa^T> - b) * 2 aa^T X
#   where (a, b) are chosen randomly.
#
#   We use the notation
#       R = <XX^T, a a^T> - b   (residual)
#       S = sign(R)
#------------------------------------------------------------------------------------------------------------

# Update residuals
function residuals!(R, S, X, a, b)
    BLAS.gemm!('N','T',1.0,X,X,0.0,R); # need to fix this
    BLAS.axpy!(-1.0,b,R);
end

# Update residual signs
function signs!(S, R)
    S = map!(sign,S,R)
end

# Update the subgradient - need to fix this
#       G(X) = S * 2 aa^T X
function subgrad!(G, S, X, a)
    BLAS.gemv!('N',1.0,S,θ,0.0,G);
    BLAS.gemv!('T',1.0,S,θ,1.0,G);
end
