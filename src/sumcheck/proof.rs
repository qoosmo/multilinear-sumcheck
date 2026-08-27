//! Sumcheck proof transcript types.
//!
//! This module defines the two data types that constitute a Sumcheck proof:
//!
//! - [`RoundPoly`]     — the degree-1 polynomial `s_j(X_j) = a + b·X_j`
//!   sent by the prover at each round.
//! - [`SumcheckProof`] — the complete transcript: claimed sum `h` plus
//!   the `n` round polynomials `s_1, …, s_n`.
//!
//! # Protocol recap
//!
//! The prover wants to convince the verifier that
//! ```text
//! h = H(f) = Σ_{x ∈ {0,1}^n} f(x).
//! ```
//!
//! At round `j` the prover sends `s_j(X_j)` represented as two field
//! elements `(a, b)` where:
//! ```text
//! s_j(X_j) = a + b · X_j
//! ```
//!
//! The verifier checks `s_j(0) + s_j(1) = s_{j-1}(r_{j-1})` and then
//! samples a fresh challenge `r_j`.
//!
//! In the tree notation used by this crate:
//! - **Canonical basis:** `s_j(X_j) = h_2' + h_3' · X_j`
//! - **Lagrange basis:**  the same affine form is reconstructed from
//!   `s_j(0)` and `s_j(1)`, using the Lagrange recurrence.
//!
//! In both cases the round polynomial is degree-1 and is fully described
//! by two field elements.

use ark_ff::Field;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// RoundPoly
// ─────────────────────────────────────────────────────────────────────────────

/// A degree-1 univariate polynomial `s(X) = a + b · X`.
///
/// This is the message the prover sends in each round of the Sumcheck
/// protocol.  Since `f` is multilinear (degree at most 1 in each
/// variable), every round polynomial has degree at most 1.
///
/// # Tree notation
///
/// In the canonical tree representation,
/// `s_j(X_j) = h_2' + h_3' · X_j`, so `a = h_2'` and `b = h_3'`.
///
/// # Encoding
///
/// Two field elements suffice to transmit `s_j` over the wire:
/// `s_j(0) = a` and `s_j(1) = a + b`.
/// The verifier can recover `a` and `b` from these two evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundPoly<F: Field> {
    /// Constant term: `s(0) = a`.
    pub a: F,
    /// Linear coefficient: `s(1) − s(0) = b`.
    pub b: F,
}

impl<F: Field> RoundPoly<F> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Construct directly from the two coefficients `a` and `b`.
    ///
    /// `s(X) = a + b · X`
    #[inline]
    pub fn new(a: F, b: F) -> Self {
        Self { a, b }
    }

    /// Construct from the two evaluations `s(0)` and `s(1)`.
    ///
    /// This is the natural encoding used in the protocol:
    /// the prover sends `(s(0), s(1))` and the verifier reconstructs.
    #[inline]
    pub fn from_evaluations(s0: F, s1: F) -> Self {
        Self { a: s0, b: s1 - s0 }
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Evaluate `s(r) = a + b · r`.
    #[inline]
    pub fn eval(&self, r: F) -> F {
        self.a + self.b * r
    }

    /// `s(0) = a`.
    #[inline]
    pub fn eval_at_zero(&self) -> F {
        self.a
    }

    /// `s(1) = a + b`.
    #[inline]
    pub fn eval_at_one(&self) -> F {
        self.a + self.b
    }

    /// `s(0) + s(1) = 2a + b`.
    ///
    /// This is the value the verifier checks against the previous round:
    /// `s_j(0) + s_j(1) = s_{j-1}(r_{j-1})`.
    #[inline]
    pub fn sum_over_boolean(&self) -> F {
        self.a + self.a + self.b
    }
}

impl<F: Field + fmt::Display> fmt::Display for RoundPoly<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {} · X", self.a, self.b)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SumcheckProof
// ─────────────────────────────────────────────────────────────────────────────

/// The complete Sumcheck proof transcript.
///
/// A `SumcheckProof` is produced by the prover and verified by the
/// verifier.  It contains:
///
/// 1. The **claimed sum** `h = H(f)`.
/// 2. The **`n` round polynomials** `s_1, …, s_n`, each of degree 1.
///
/// # Size
///
/// A proof for a polynomial in `n` variables consists of `2n + 1` field
/// elements: one for the claimed sum, and two per round polynomial.
///
/// # Usage
///
/// ```text
/// // Prover side
/// let proof = prover.prove(&challenges);
///
/// // Verifier side
/// let oracle_eval = f.evaluate(&challenges);
/// verifier.verify(&proof, &challenges, oracle_eval)?;
/// ```
#[derive(Debug, Clone)]
pub struct SumcheckProof<F: Field> {
    /// The claimed hypercube sum `h = H(f) = Σ_{x ∈ {0,1}^n} f(x)`.
    ///
    /// This is the first value the prover sends.
    pub claimed_sum: F,

    /// The `n` round polynomials `s_1, …, s_n`.
    ///
    /// `round_polys[j-1]` = `s_j` (0-based storage, 1-based round index).
    pub round_polys: Vec<RoundPoly<F>>,
}

impl<F: Field> SumcheckProof<F> {
    /// Construct a proof from a claimed sum and a vector of round polynomials.
    pub fn new(claimed_sum: F, round_polys: Vec<RoundPoly<F>>) -> Self {
        Self {
            claimed_sum,
            round_polys,
        }
    }

    /// Number of variables `n` — equals the number of rounds.
    pub fn num_vars(&self) -> usize {
        self.round_polys.len()
    }

    /// Total number of field elements in the proof: `2n + 1`.
    pub fn size_in_field_elements(&self) -> usize {
        1 + 2 * self.round_polys.len()
    }

    /// The round polynomial `s_j` using the **1-based round index**.
    ///
    /// # Panics
    /// Panics if `j == 0` or `j > n`.
    pub fn round_poly(&self, j: usize) -> &RoundPoly<F> {
        assert!(
            j >= 1 && j <= self.round_polys.len(),
            "round index {j} out of range [1, {}]",
            self.round_polys.len()
        );
        &self.round_polys[j - 1]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    // ── RoundPoly construction ────────────────────────────────────────────────

    #[test]
    fn new_stores_a_and_b() {
        let s = RoundPoly::new(fr(3), fr(5));
        assert_eq!(s.a, fr(3));
        assert_eq!(s.b, fr(5));
    }

    #[test]
    fn from_evaluations_recovers_a_and_b() {
        // s(0)=3, s(1)=8  →  a=3, b=5
        let s = RoundPoly::from_evaluations(fr(3), fr(8));
        assert_eq!(s.a, fr(3));
        assert_eq!(s.b, fr(5));
    }

    #[test]
    fn from_evaluations_roundtrip() {
        let s0 = fr(7);
        let s1 = fr(13);
        let s = RoundPoly::from_evaluations(s0, s1);
        assert_eq!(s.eval_at_zero(), s0);
        assert_eq!(s.eval_at_one(), s1);
    }

    // ── RoundPoly evaluation ──────────────────────────────────────────────────

    #[test]
    fn eval_at_zero_returns_a() {
        let s = RoundPoly::new(fr(4), fr(7));
        assert_eq!(s.eval(fr(0)), fr(4));
        assert_eq!(s.eval_at_zero(), fr(4));
    }

    #[test]
    fn eval_at_one_returns_a_plus_b() {
        let s = RoundPoly::new(fr(4), fr(7));
        assert_eq!(s.eval(fr(1)), fr(11));
        assert_eq!(s.eval_at_one(), fr(11));
    }

    #[test]
    fn eval_at_arbitrary_r() {
        // s(X) = 2 + 3X  →  s(5) = 2 + 15 = 17
        let s = RoundPoly::new(fr(2), fr(3));
        assert_eq!(s.eval(fr(5)), fr(17));
    }

    #[test]
    fn sum_over_boolean_is_s0_plus_s1() {
        // s(X) = 4 + 7X  →  s(0)+s(1) = 4 + 11 = 15 = 2*4+7
        let s = RoundPoly::new(fr(4), fr(7));
        assert_eq!(s.sum_over_boolean(), s.eval_at_zero() + s.eval_at_one());
        assert_eq!(s.sum_over_boolean(), fr(15));
    }

    #[test]
    fn sum_over_boolean_formula_2a_plus_b() {
        // sum_over_boolean = 2a + b
        let s = RoundPoly::new(fr(3), fr(5));
        assert_eq!(s.sum_over_boolean(), fr(2) * fr(3) + fr(5));
    }

    // ── RoundPoly zero polynomial ─────────────────────────────────────────────

    #[test]
    fn zero_round_poly_evals_to_zero() {
        use ark_ff::Zero;
        let s = RoundPoly::new(Fr::zero(), Fr::zero());
        assert_eq!(s.eval(fr(42)), Fr::zero());
        assert_eq!(s.sum_over_boolean(), Fr::zero());
    }

    // ── SumcheckProof ─────────────────────────────────────────────────────────

    #[test]
    fn proof_num_vars_equals_round_count() {
        let polys = vec![
            RoundPoly::new(fr(1), fr(2)),
            RoundPoly::new(fr(3), fr(4)),
            RoundPoly::new(fr(5), fr(6)),
        ];
        let proof = SumcheckProof::new(fr(42), polys);
        assert_eq!(proof.num_vars(), 3);
    }

    #[test]
    fn proof_size_is_2n_plus_1() {
        let polys = vec![RoundPoly::new(fr(1), fr(2)), RoundPoly::new(fr(3), fr(4))];
        let proof = SumcheckProof::new(fr(10), polys);
        // n=2: size = 2*2+1 = 5
        assert_eq!(proof.size_in_field_elements(), 5);
    }

    #[test]
    fn proof_round_poly_1based_accessor() {
        let s1 = RoundPoly::new(fr(1), fr(2));
        let s2 = RoundPoly::new(fr(3), fr(4));
        let proof = SumcheckProof::new(fr(0), vec![s1, s2]);
        assert_eq!(*proof.round_poly(1), s1);
        assert_eq!(*proof.round_poly(2), s2);
    }

    #[test]
    #[should_panic]
    fn proof_round_poly_index_zero_panics() {
        let proof = SumcheckProof::new(fr(0), vec![RoundPoly::new(fr(1), fr(2))]);
        proof.round_poly(0);
    }

    #[test]
    #[should_panic]
    fn proof_round_poly_out_of_range_panics() {
        let proof = SumcheckProof::new(fr(0), vec![RoundPoly::new(fr(1), fr(2))]);
        proof.round_poly(2);
    }

    // ── Verifier check simulation ─────────────────────────────────────────────

    /// Simulate the verifier's round checks manually.
    ///
    /// For a 2-round proof with known values, verify that
    /// `s_1(0) + s_1(1) = claimed_sum` and
    /// `s_2(0) + s_2(1) = s_1(r_1)`.
    #[test]
    fn verifier_round_check_simulation() {
        // f = 1 + 2x₁ + 3x₂ + 4x₁x₂, H(f) = 18
        // Round 1: s_1(X) = 6 + 12·X  (h_2=6, h_3=12 from the h-tree)
        //   s_1(0) + s_1(1) = 6 + 18 = 24 ... but H=18
        // Let's use a simpler consistent example:
        // s_1(X) = a + b·X with a + (a+b) = 18 → 2a+b = 18
        // Choose a=5, b=8: 10+8=18 ✓
        // r_1 = 3: s_1(3) = 5 + 24 = 29
        // s_2(X) = c + d·X with 2c+d = 29
        // Choose c=10, d=9: 20+9=29 ✓
        let s1 = RoundPoly::new(fr(5), fr(8));
        let s2 = RoundPoly::new(fr(10), fr(9));
        let proof = SumcheckProof::new(fr(18), vec![s1, s2]);

        let r1 = fr(3);

        // Round 1 check: s_1(0) + s_1(1) = claimed_sum
        assert_eq!(proof.round_poly(1).sum_over_boolean(), proof.claimed_sum);

        // Round 2 check: s_2(0) + s_2(1) = s_1(r_1)
        assert_eq!(
            proof.round_poly(2).sum_over_boolean(),
            proof.round_poly(1).eval(r1)
        );
    }
}
