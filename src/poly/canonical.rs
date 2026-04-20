use ark_ff::Field;

use super::traits::MlPoly;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalTerm<F: Field> {
    pub coeff: F,
    pub index: usize,
}

impl<F: Field> CanonicalTerm<F> {
    pub fn new(coeff: F, index: usize) -> Self {
        Self { coeff, index }
    }

    pub fn degree(&self) -> u32 {
        self.index.count_ones()
    }

    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    pub fn eval_boolean(&self, b: usize) -> F {
        if self.index & b == self.index {
            F::one()
        } else {
            F::zero()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPoly<F: Field> {
    num_vars: usize,
    coeffs: Vec<F>,
}

impl<F: Field> CanonicalPoly<F> {
    pub fn new(coeffs: Vec<F>) -> Self {
        let n = coeffs.len();
        assert!(
            n.is_power_of_two() && n > 0,
            "CanonicalPoly: length must be a power of two, got {n}"
        );
        Self {
            num_vars: n.trailing_zeros() as usize,
            coeffs,
        }
    }

    pub fn zero(num_vars: usize) -> Self {
        Self {
            num_vars,
            coeffs: vec![F::zero(); 1 << num_vars],
        }
    }

    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut [F] {
        &mut self.coeffs
    }

    pub fn coeff(&self, j: usize) -> F {
        self.coeffs[j]
    }

    pub fn terms(&self) -> impl Iterator<Item = CanonicalTerm<F>> + '_ {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(j, &coeff)| CanonicalTerm::new(coeff, j))
    }

    pub fn all_terms(&self) -> impl Iterator<Item = CanonicalTerm<F>> + '_ {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(j, &coeff)| CanonicalTerm::new(coeff, j))
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Naive evaluation at `r = (r₁, …, rₙ)`.
    ///
    /// Computes `f(r) = Σⱼ αⱼ · mⱼ(r)` by evaluating each monomial
    /// individually and summing.
    ///
    /// `mⱼ(r) = r₁^{j₁} · r₂^{j₂} · … · rₙ^{jₙ}`
    ///
    /// Since `jₖ ∈ {0,1}` this simplifies to: multiply together the `rₖ`
    /// for which bit `k` of `j` is set.
    ///
    /// # Complexity
    /// O(N log N) — proportional to the total degree of all monomials.
    ///
    /// # Panics
    /// Panics if `r.len() != num_vars`.
    pub fn eval_naive(&self, r: &[F]) -> F {
        assert_eq!(
            r.len(), self.num_vars,
            "eval_naive: expected {} variables, got {}", self.num_vars, r.len()
        );

        self.coeffs
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, &alpha)| {
                if alpha.is_zero() {
                    return acc;
                }
                // Evaluate mⱼ(r): multiply rₖ for every bit k set in j.
                let monomial_val = (0..self.num_vars).fold(F::one(), |prod, k| {
                    if (j >> k) & 1 == 1 {
                        prod * r[k]
                    } else {
                        prod
                    }
                });
                acc + alpha * monomial_val
            })
    }

    /// Circuit-based evaluation at `r = (r₁, …, rₙ)`.
    ///
    /// Implements the bottom-up traversal of the `(q_j)` tree from §3.3
    /// of the paper.
    ///
    /// # Algorithm
    ///
    /// Start from a working buffer initialised with the leaf values
    /// (the coefficient vector itself, since `q_j` at the leaf layer
    /// holds individual coefficients).
    ///
    /// At each layer `i` (from `n` down to `1`), apply:
    /// ```text
    /// buf[t] ← buf[2t] + r_i · buf[2t+1]   for t = 0 .. half
    /// ```
    /// After `n` layers the buffer has a single entry: `f(r)`.
    ///
    /// # Complexity
    /// Exactly `N − 1` multiplications and `N − 1` additions — O(N).
    ///
    /// # Panics
    /// Panics if `r.len() != num_vars`.
    pub fn eval_circuit(&self, r: &[F]) -> F {
    assert_eq!(
        r.len(), self.num_vars,
        "eval_circuit: expected {} variables, got {}", self.num_vars, r.len()
    );

    // Working buffer: starts as a copy of the coefficient vector.
    // We fold variable x₁ first (r[0]), then x₂ (r[1]), …, xₙ (r[n-1]).
    // This matches the split rule: q₂ holds even-indexed coefficients
    // (j₁ = 0) and q₃ holds odd-indexed coefficients (j₁ = 1).
    let mut buf = self.coeffs.clone();

    for k in 0..self.num_vars {
        let r_k  = r[k];
        let half = buf.len() / 2;
        for t in 0..half {
            buf[t] = buf[2 * t] + r_k * buf[2 * t + 1];
        }
        buf.truncate(half);
    }

    buf[0]
}
}

// ── MlPoly implementation ─────────────────────────────────────────────────────

impl<F: Field> MlPoly<F> for CanonicalPoly<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn hypercube_sum(&self) -> F {
        let n = self.num_vars;
        self.coeffs
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (j, &alpha)| {
                if alpha.is_zero() {
                    acc
                } else {
                    let deg    = j.count_ones() as usize;
                    let weight = F::from(1u64 << (n - deg));
                    acc + alpha * weight
                }
            })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::One;
    use ark_ff::Zero;

    fn fr(n: u64) -> Fr { Fr::from(n) }

    // ── CanonicalTerm ─────────────────────────────────────────────────────────

    #[test]
    fn term_degree_is_hamming_weight() {
        assert_eq!(CanonicalTerm::<Fr>::new(fr(1), 0).degree(), 0);
        assert_eq!(CanonicalTerm::<Fr>::new(fr(1), 1).degree(), 1);
        assert_eq!(CanonicalTerm::<Fr>::new(fr(1), 3).degree(), 2);
        assert_eq!(CanonicalTerm::<Fr>::new(fr(1), 7).degree(), 3);
    }

    #[test]
    fn term_eval_boolean_correct() {
        let t = CanonicalTerm::<Fr>::new(fr(1), 3);
        assert_eq!(t.eval_boolean(3), Fr::one());
        assert_eq!(t.eval_boolean(7), Fr::one());
        assert_eq!(t.eval_boolean(1), Fr::zero());
        assert_eq!(t.eval_boolean(2), Fr::zero());
    }

    #[test]
    fn term_zero_detection() {
        assert!(CanonicalTerm::<Fr>::new(Fr::zero(), 5).is_zero());
        assert!(!CanonicalTerm::<Fr>::new(fr(1), 5).is_zero());
    }

    // ── CanonicalPoly construction ────────────────────────────────────────────

    #[test]
    fn new_sets_num_vars_correctly() {
        let p = CanonicalPoly::new(vec![fr(0); 4]);
        assert_eq!(p.num_vars(), 2);
        assert_eq!(p.num_evals(), 4);
    }

    #[test]
    fn zero_poly_has_all_zero_coeffs() {
        let p = CanonicalPoly::<Fr>::zero(3);
        assert!(p.coeffs().iter().all(|c| c.is_zero()));
    }

    #[test]
    #[should_panic]
    fn new_panics_on_non_power_of_two() {
        CanonicalPoly::<Fr>::new(vec![fr(1); 3]);
    }

    // ── terms iterator ────────────────────────────────────────────────────────

    #[test]
    fn terms_skips_zero_coefficients() {
        let p = CanonicalPoly::new(vec![fr(5), fr(0), fr(3), fr(0)]);
        let terms: Vec<_> = p.terms().collect();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0], CanonicalTerm::new(fr(5), 0));
        assert_eq!(terms[1], CanonicalTerm::new(fr(3), 2));
    }

    #[test]
    fn all_terms_has_correct_length() {
        let p = CanonicalPoly::<Fr>::zero(3);
        assert_eq!(p.all_terms().count(), 8);
    }

    // ── hypercube_sum ─────────────────────────────────────────────────────────

    #[test]
    fn hypercube_sum_constant_poly() {
        let mut coeffs = vec![Fr::zero(); 8];
        coeffs[0] = fr(7);
        let p = CanonicalPoly::new(coeffs);
        assert_eq!(p.hypercube_sum(), fr(56));
    }

    #[test]
    fn hypercube_sum_linear_term() {
        let mut coeffs = vec![Fr::zero(); 8];
        coeffs[1] = fr(1);
        let p = CanonicalPoly::new(coeffs);
        assert_eq!(p.hypercube_sum(), fr(4));
    }

    #[test]
    fn hypercube_sum_full_degree_term() {
        let mut coeffs = vec![Fr::zero(); 8];
        coeffs[7] = fr(1);
        let p = CanonicalPoly::new(coeffs);
        assert_eq!(p.hypercube_sum(), fr(1));
    }

    #[test]
    fn hypercube_sum_matches_direct_sum_n2() {
        let p = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        assert_eq!(p.hypercube_sum(), fr(18));
    }

    #[test]
    fn hypercube_sum_zero_poly() {
        let p = CanonicalPoly::<Fr>::zero(4);
        assert_eq!(p.hypercube_sum(), Fr::zero());
    }

    // ── eval_naive ────────────────────────────────────────────────────────────

    /// f = 1 + 2x₁ + 3x₂ + 4x₁x₂
    /// f(0,0)=1, f(1,0)=3, f(0,1)=4, f(1,1)=10
    #[test]
    fn eval_naive_at_boolean_points_n2() {
        let f = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        assert_eq!(f.eval_naive(&[fr(0), fr(0)]), fr(1));
        assert_eq!(f.eval_naive(&[fr(1), fr(0)]), fr(3));
        assert_eq!(f.eval_naive(&[fr(0), fr(1)]), fr(4));
        assert_eq!(f.eval_naive(&[fr(1), fr(1)]), fr(10));
    }

    #[test]
    fn eval_naive_constant_poly() {
        // f = 5  (only α₀ = 5)
        let mut coeffs = vec![Fr::zero(); 4];
        coeffs[0] = fr(5);
        let f = CanonicalPoly::new(coeffs);
        assert_eq!(f.eval_naive(&[fr(3), fr(7)]), fr(5));
    }

    #[test]
    fn eval_naive_single_variable_n1() {
        // f = 3 + 7x₁
        // f(0)=3, f(1)=10, f(2)=3+14=17
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        assert_eq!(f.eval_naive(&[fr(0)]), fr(3));
        assert_eq!(f.eval_naive(&[fr(1)]), fr(10));
        assert_eq!(f.eval_naive(&[fr(2)]), fr(17));
    }

    #[test]
    #[should_panic]
    fn eval_naive_wrong_point_length_panics() {
        let f = CanonicalPoly::new(vec![fr(1); 4]);
        f.eval_naive(&[fr(1)]);
    }

    // ── eval_circuit ──────────────────────────────────────────────────────────

    #[test]
    fn eval_circuit_at_boolean_points_n2() {
        let f = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        assert_eq!(f.eval_circuit(&[fr(0), fr(0)]), fr(1));
        assert_eq!(f.eval_circuit(&[fr(1), fr(0)]), fr(3));
        assert_eq!(f.eval_circuit(&[fr(0), fr(1)]), fr(4));
        assert_eq!(f.eval_circuit(&[fr(1), fr(1)]), fr(10));
    }

    #[test]
    fn eval_circuit_constant_poly() {
        let mut coeffs = vec![Fr::zero(); 4];
        coeffs[0] = fr(5);
        let f = CanonicalPoly::new(coeffs);
        assert_eq!(f.eval_circuit(&[fr(3), fr(7)]), fr(5));
    }

    #[test]
    fn eval_circuit_single_variable_n1() {
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        assert_eq!(f.eval_circuit(&[fr(0)]), fr(3));
        assert_eq!(f.eval_circuit(&[fr(1)]), fr(10));
        assert_eq!(f.eval_circuit(&[fr(2)]), fr(17));
    }

    #[test]
    #[should_panic]
    fn eval_circuit_wrong_point_length_panics() {
        let f = CanonicalPoly::new(vec![fr(1); 4]);
        f.eval_circuit(&[fr(1)]);
    }

    // ── Cross-checks: naive and circuit must agree ────────────────────────────

    #[test]
    fn naive_and_circuit_agree_at_boolean_points_n3() {
        // f = α₀ + α₁x₁ + … + α₇x₁x₂x₃
        let coeffs: Vec<Fr> = (1..=8).map(|i| fr(i)).collect();
        let f = CanonicalPoly::new(coeffs);
        for b0 in [fr(0), fr(1)] {
            for b1 in [fr(0), fr(1)] {
                for b2 in [fr(0), fr(1)] {
                    let r = [b0, b1, b2];
                    assert_eq!(
                        f.eval_naive(&r),
                        f.eval_circuit(&r),
                        "disagreement at r={:?}", r
                    );
                }
            }
        }
    }

    #[test]
    fn naive_and_circuit_agree_at_random_point_n3() {
        let coeffs: Vec<Fr> = (1..=8).map(|i| fr(i)).collect();
        let f = CanonicalPoly::new(coeffs);
        // Use a fixed non-boolean point to test the general case.
        let r = [fr(2), fr(5), fr(11)];
        assert_eq!(f.eval_naive(&r), f.eval_circuit(&r));
    }

    #[test]
    fn naive_and_circuit_agree_n4() {
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 3 + 1)).collect();
        let f = CanonicalPoly::new(coeffs);
        let r = [fr(3), fr(7), fr(13), fr(2)];
        assert_eq!(f.eval_naive(&r), f.eval_circuit(&r));
    }

    // ── eval_circuit buffer size after each fold ──────────────────────────────

    /// The circuit evaluation must return the same value whether we fold
    /// variables in forward or reverse order — but our implementation
    /// always folds x_n first (right-to-left), matching the paper's
    /// canonical order x₁ → x₂ → … → xₙ for the decomposition tree.
    #[test]
    fn eval_circuit_zero_poly_is_zero() {
        let f = CanonicalPoly::<Fr>::zero(4);
        let r = [fr(1), fr(2), fr(3), fr(4)];
        assert_eq!(f.eval_circuit(&r), Fr::zero());
    }
}