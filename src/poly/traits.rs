use ark_ff::Field;

/// Common interface for a multilinear polynomial in `n` variables over `F`.
///
/// Implemented by both [`CanonicalPoly`] and [`LagrangePoly`].
/// Every algorithm in the circuit and sumcheck layers depends only on this
/// trait, never on a concrete type.
pub trait MlPoly<F: Field> {
    /// Number of variables `n`.
    fn num_vars(&self) -> usize;

    /// `N = 2^n` — length of the underlying vector.
    fn num_evals(&self) -> usize {
        1 << self.num_vars()
    }

    /// The Boolean hypercube sum `H(f) = Σ_{x ∈ {0,1}^n} f(x)`.
    fn hypercube_sum(&self) -> F;
}
