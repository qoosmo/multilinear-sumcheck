use ark_ff::Field;
use rayon::prelude::*;

use super::traits::MlPoly;

/// A multilinear polynomial in the Lagrange (evaluation) basis.
///
/// Stored as a dense evaluation vector `(f(0), f(1), …, f(N-1))` of length
/// `N = 2^n`, where entry `j` is the value of `f` at the Boolean point
/// `(j₁, j₂, …, jₙ)` defined by the binary decomposition of `j`
/// (`j₁` is the least significant bit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LagrangePoly<F: Field> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F: Field> LagrangePoly<F> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Construct from a dense evaluation vector.
    ///
    /// # Panics
    /// Panics if `evals.len()` is not a power of two, or is zero.
    pub fn new(evals: Vec<F>) -> Self {
        let n = evals.len();
        assert!(
            n.is_power_of_two() && n > 0,
            "LagrangePoly: length must be a power of two, got {n}"
        );
        Self {
            num_vars: n.trailing_zeros() as usize,
            evals,
        }
    }

    /// Construct the zero polynomial in `n` variables.
    pub fn zero(num_vars: usize) -> Self {
        Self {
            num_vars,
            evals: vec![F::zero(); 1 << num_vars],
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// The evaluation vector as a slice.
    pub fn evals(&self) -> &[F] {
        &self.evals
    }

    /// Mutable access to the evaluation vector.
    pub fn evals_mut(&mut self) -> &mut [F] {
        &mut self.evals
    }

    /// The value `f(j₁, …, jₙ)` at the Boolean point encoded by `j`.
    ///
    /// # Panics
    /// Panics if `j >= N`.
    pub fn eval_at_boolean(&self, j: usize) -> F {
        self.evals[j]
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Standard evaluation at `r = (r₁, …, rₙ) ∈ 𝔽ⁿ`.
    ///
    /// Fold step: `(1 - r) · u  +  r · v`
    /// Cost per pair: 2 multiplications, 1 addition.
    ///
    /// # Panics
    /// Panics if `r.len() != num_vars`.
    pub fn eval_standard(&self, r: &[F]) -> F {
        assert_eq!(
            r.len(),
            self.num_vars,
            "eval_standard: expected {} variables, got {}",
            self.num_vars,
            r.len()
        );

        let mut buf = self.evals.clone();

        for k in 0..self.num_vars {
            let r_k = r[k];
            let one_minus_r = F::one() - r_k;
            let half = buf.len() / 2;
            for t in 0..half {
                buf[t] = one_minus_r * buf[2 * t] + r_k * buf[2 * t + 1];
            }
            buf.truncate(half);
        }

        buf[0]
    }

    /// Optimized sequential evaluation at `r = (r₁, …, rₙ) ∈ 𝔽ⁿ`.
    ///
    /// Fold step: `u  +  r · (v - u)`
    /// Cost per pair: 1 multiplication, 2 additions.
    ///
    /// Saves one multiplication per pair vs `eval_standard`.
    /// Paper reports ~25–35% improvement over `eval_standard`.
    ///
    /// # Panics
    /// Panics if `r.len() != num_vars`.
    pub fn eval_optimized(&self, r: &[F]) -> F {
        assert_eq!(
            r.len(),
            self.num_vars,
            "eval_optimized: expected {} variables, got {}",
            self.num_vars,
            r.len()
        );

        let mut buf = self.evals.clone();

        for k in 0..self.num_vars {
            let r_k = r[k];
            let half = buf.len() / 2;
            for t in 0..half {
                buf[t] = buf[2 * t] + r_k * (buf[2 * t + 1] - buf[2 * t]);
            }
            buf.truncate(half);
        }

        buf[0]
    }

    /// Parallel optimized evaluation at `r = (r₁, …, rₙ) ∈ 𝔽ⁿ`.
    ///
    /// Same fold step as `eval_optimized`: `u + r · (v - u)`
    /// but the inner loop over pairs is parallelized using `rayon`.
    ///
    /// # Why this works
    ///
    /// At each layer `k`, every pair `(buf[2t], buf[2t+1])` is independent
    /// of every other pair. There are no data dependencies across pairs.
    /// This is exactly the condition rayon requires for safe parallelism.
    ///
    /// # When this is faster
    ///
    /// The parallel version is faster than `eval_optimized` only when
    /// the per-layer work is large enough to amortise thread synchronisation.
    /// In practice this means `n ≥ 18` on most machines.
    /// At `n < 16` the sequential version is typically faster.
    ///
    /// # Panics
    /// Panics if `r.len() != num_vars`.
    pub fn eval_parallel(&self, r: &[F]) -> F
    where
        F: Send + Sync,
    {
        assert_eq!(
            r.len(),
            self.num_vars,
            "eval_parallel: expected {} variables, got {}",
            self.num_vars,
            r.len()
        );

        let mut buf = self.evals.clone();
        // Pre-allocate a reusable buffer — avoids one allocation per layer.
        let mut tmp = Vec::with_capacity(buf.len());

        for k in 0..self.num_vars {
            let r_k = r[k];
            tmp.clear();
            buf.par_chunks(2)
                .map(|pair| pair[0] + r_k * (pair[1] - pair[0]))
                .collect_into_vec(&mut tmp);
            std::mem::swap(&mut buf, &mut tmp);
        }

        buf[0]
    }
}

// ── MlPoly implementation ─────────────────────────────────────────────────────

impl<F: Field> MlPoly<F> for LagrangePoly<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// `H(f) = Σ_{j=0}^{N-1} f(j)`
    fn hypercube_sum(&self) -> F {
        self.evals.iter().fold(F::zero(), |acc, &v| acc + v)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_sets_num_vars_correctly() {
        let p = LagrangePoly::new(vec![fr(0); 8]);
        assert_eq!(p.num_vars(), 3);
        assert_eq!(p.num_evals(), 8);
    }

    #[test]
    fn zero_poly_has_all_zero_evals() {
        let p = LagrangePoly::<Fr>::zero(3);
        assert!(p.evals().iter().all(|e| e.is_zero()));
    }

    #[test]
    #[should_panic]
    fn new_panics_on_non_power_of_two() {
        LagrangePoly::<Fr>::new(vec![fr(1); 5]);
    }

    // ── eval_at_boolean ───────────────────────────────────────────────────────

    #[test]
    fn eval_at_boolean_returns_correct_entry() {
        let p = LagrangePoly::new(vec![fr(10), fr(20), fr(30), fr(40)]);
        assert_eq!(p.eval_at_boolean(0), fr(10));
        assert_eq!(p.eval_at_boolean(1), fr(20));
        assert_eq!(p.eval_at_boolean(2), fr(30));
        assert_eq!(p.eval_at_boolean(3), fr(40));
    }

    // ── hypercube_sum ─────────────────────────────────────────────────────────

    #[test]
    fn hypercube_sum_is_sum_of_evals() {
        let p = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        assert_eq!(p.hypercube_sum(), fr(18));
    }

    #[test]
    fn hypercube_sum_zero_poly() {
        let p = LagrangePoly::<Fr>::zero(4);
        assert_eq!(p.hypercube_sum(), Fr::zero());
    }

    #[test]
    fn hypercube_sum_matches_canonical_n2() {
        let p = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        assert_eq!(p.hypercube_sum(), fr(18));
    }

    // ── eval_standard ─────────────────────────────────────────────────────────

    #[test]
    fn eval_standard_at_boolean_points_n2() {
        let p = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        assert_eq!(p.eval_standard(&[fr(0), fr(0)]), fr(1));
        assert_eq!(p.eval_standard(&[fr(1), fr(0)]), fr(3));
        assert_eq!(p.eval_standard(&[fr(0), fr(1)]), fr(4));
        assert_eq!(p.eval_standard(&[fr(1), fr(1)]), fr(10));
    }

    #[test]
    fn eval_standard_n1() {
        let p = LagrangePoly::new(vec![fr(3), fr(10)]);
        assert_eq!(p.eval_standard(&[fr(0)]), fr(3));
        assert_eq!(p.eval_standard(&[fr(1)]), fr(10));
        assert_eq!(p.eval_standard(&[fr(2)]), fr(17));
    }

    #[test]
    fn eval_standard_zero_poly() {
        let p = LagrangePoly::<Fr>::zero(3);
        assert_eq!(p.eval_standard(&[fr(1), fr(2), fr(3)]), Fr::zero());
    }

    #[test]
    #[should_panic]
    fn eval_standard_wrong_length_panics() {
        let p = LagrangePoly::new(vec![fr(1); 4]);
        p.eval_standard(&[fr(1)]);
    }

    // ── eval_optimized ────────────────────────────────────────────────────────

    #[test]
    fn eval_optimized_at_boolean_points_n2() {
        let p = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        assert_eq!(p.eval_optimized(&[fr(0), fr(0)]), fr(1));
        assert_eq!(p.eval_optimized(&[fr(1), fr(0)]), fr(3));
        assert_eq!(p.eval_optimized(&[fr(0), fr(1)]), fr(4));
        assert_eq!(p.eval_optimized(&[fr(1), fr(1)]), fr(10));
    }

    #[test]
    fn eval_optimized_n1() {
        let p = LagrangePoly::new(vec![fr(3), fr(10)]);
        assert_eq!(p.eval_optimized(&[fr(0)]), fr(3));
        assert_eq!(p.eval_optimized(&[fr(1)]), fr(10));
        assert_eq!(p.eval_optimized(&[fr(2)]), fr(17));
    }

    #[test]
    fn eval_optimized_zero_poly() {
        let p = LagrangePoly::<Fr>::zero(3);
        assert_eq!(p.eval_optimized(&[fr(1), fr(2), fr(3)]), Fr::zero());
    }

    #[test]
    #[should_panic]
    fn eval_optimized_wrong_length_panics() {
        let p = LagrangePoly::new(vec![fr(1); 4]);
        p.eval_optimized(&[fr(1)]);
    }

    // ── eval_parallel ─────────────────────────────────────────────────────────

    #[test]
    fn eval_parallel_at_boolean_points_n2() {
        let p = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        assert_eq!(p.eval_parallel(&[fr(0), fr(0)]), fr(1));
        assert_eq!(p.eval_parallel(&[fr(1), fr(0)]), fr(3));
        assert_eq!(p.eval_parallel(&[fr(0), fr(1)]), fr(4));
        assert_eq!(p.eval_parallel(&[fr(1), fr(1)]), fr(10));
    }

    #[test]
    fn eval_parallel_n1() {
        let p = LagrangePoly::new(vec![fr(3), fr(10)]);
        assert_eq!(p.eval_parallel(&[fr(0)]), fr(3));
        assert_eq!(p.eval_parallel(&[fr(1)]), fr(10));
        assert_eq!(p.eval_parallel(&[fr(2)]), fr(17));
    }

    #[test]
    fn eval_parallel_zero_poly() {
        let p = LagrangePoly::<Fr>::zero(3);
        assert_eq!(p.eval_parallel(&[fr(1), fr(2), fr(3)]), Fr::zero());
    }

    // ── All three agree at all points ─────────────────────────────────────────

    #[test]
    fn all_three_agree_at_boolean_points_n3() {
        let p = LagrangePoly::new((1..=8).map(|i| fr(i)).collect());
        for b0 in [fr(0), fr(1)] {
            for b1 in [fr(0), fr(1)] {
                for b2 in [fr(0), fr(1)] {
                    let r = [b0, b1, b2];
                    let s = p.eval_standard(&r);
                    let o = p.eval_optimized(&r);
                    let par = p.eval_parallel(&r);
                    assert_eq!(s, o, "standard vs optimized at {:?}", r);
                    assert_eq!(o, par, "optimized vs parallel at {:?}", r);
                }
            }
        }
    }

    #[test]
    fn all_three_agree_at_random_point_n4() {
        let p = LagrangePoly::new((0..16).map(|i| fr(i as u64 * 3 + 1)).collect());
        let r = [fr(2), fr(5), fr(11), fr(7)];
        let s = p.eval_standard(&r);
        let o = p.eval_optimized(&r);
        let par = p.eval_parallel(&r);
        assert_eq!(s, o);
        assert_eq!(o, par);
    }

    #[test]
    fn lagrange_eval_agrees_with_canonical_eval_circuit_n3() {
        use crate::circuit::LagrangeDecomp;
        use crate::poly::CanonicalPoly;
        let coeffs: Vec<Fr> = (1..=8).map(|i| fr(i)).collect();
        let canon = CanonicalPoly::new(coeffs);
        let lag = LagrangeDecomp::build(&canon).to_lagrange();
        let r = [fr(3), fr(7), fr(13)];
        assert_eq!(lag.eval_optimized(&r), canon.eval_circuit(&r));
        assert_eq!(lag.eval_parallel(&r), canon.eval_circuit(&r));
    }

    #[test]
    fn lagrange_eval_agrees_with_canonical_eval_circuit_n4() {
        use crate::circuit::LagrangeDecomp;
        use crate::poly::CanonicalPoly;
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 3 + 1)).collect();
        let canon = CanonicalPoly::new(coeffs);
        let lag = LagrangeDecomp::build(&canon).to_lagrange();
        let r = [fr(2), fr(5), fr(11), fr(7)];
        assert_eq!(lag.eval_optimized(&r), canon.eval_circuit(&r));
        assert_eq!(lag.eval_parallel(&r), canon.eval_circuit(&r));
    }
}
