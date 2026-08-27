//! Sumcheck provers — canonical and Lagrange basis.
//!
//! Both provers follow the same structure:
//!
//! 1. At construction, build the `(h_j)` sum circuit from the polynomial
//!    in `O(N)` time and store it.
//! 2. At each round `j`, read layer `j` of the stored circuit, fold with
//!    the challenges received so far, and return a [`RoundPoly`].
//! 3. Assemble the full [`SumcheckProof`] transcript.
//!
//! # Round polynomial algorithm (canonical basis, §5.4)
//!
//! At round `j` with challenges `r_1, …, r_{j-1}` already received:
//!
//! 1. Read the layer-`j` values: `h = [h_{2^j}, h_{2^j+1}, …, h_{2^{j+1}-1}]`.
//! 2. Fold using the most-recently-received challenge first:
//!    ```text
//!    h'[k] = h[2k]   + r · h[2k+2]   for k even
//!    h'[k] = h[2k-1] + r · h[2k+1]   for k odd
//!    ```
//! 3. Return `s_j(X_j) = h[0] + h[1] · X_j`.
//!
//! # Round polynomial algorithm (Lagrange basis, §5.5)
//!
//! Same fold rule but with the optimized formula:
//! ```text
//! h'[k] = h[2k]   + r · (h[2k+2] − h[2k])    for k even
//! h'[k] = h[2k-1] + r · (h[2k+1] − h[2k-1])  for k odd
//! ```
//! Returns `s_j` via `from_evaluations(h[0], h[1])` since in the Lagrange
//! basis `s_j(X) = h[0]·(1−X) + h[1]·X`.
//!
//! # Complexity (from the paper, Table 1)
//!
//! | Algorithm           | Multiplications | Additions |
//! |---------------------|----------------|-----------|
//! | LinearTimeSC        | `2N`            | `3N`      |
//! | **CanonicalProver** | **`2N`**        | **`2N`**  |
//! | **LagrangeProver**  | **`2N`**        | **`4N`**  |

use ark_ff::Field;

use crate::circuit::{CanonicalSumCircuit, LagrangeSumCircuit, SumCircuit};
use crate::poly::{CanonicalPoly, LagrangePoly};
use crate::sumcheck::proof::{RoundPoly, SumcheckProof};

// ─────────────────────────────────────────────────────────────────────────────
// CanonicalProver
// ─────────────────────────────────────────────────────────────────────────────

/// Sumcheck prover for the **canonical basis**.
pub struct CanonicalProver<F: Field> {
    circuit: CanonicalSumCircuit<F>,
}

impl<F: Field> CanonicalProver<F> {
    /// Build the prover from a canonical polynomial. Cost: `O(N)`.
    pub fn new(f: &CanonicalPoly<F>) -> Self {
        Self {
            circuit: CanonicalSumCircuit::build(f),
        }
    }

    /// The claimed sum `H(f)` — the first value sent to the verifier.
    #[inline]
    pub fn claimed_sum(&self) -> F {
        self.circuit.root()
    }

    /// Number of variables `n`.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.circuit.num_vars()
    }

    /// Compute `s_j(X_j)` given challenges `r_1, …, r_{j-1}`.
    ///
    /// Pass an empty slice for round 1.
    ///
    /// # Panics
    /// Panics if `challenges.len() >= num_vars`.
    pub fn compute_round_poly(&self, challenges: &[F]) -> RoundPoly<F> {
        let j = challenges.len() + 1;
        debug_assert!(j <= self.circuit.num_vars());

        // Read layer j: 1-based indices [2^j, 2^{j+1} − 1].
        let layer_start = 1usize << j;
        let layer_size = 1usize << j;
        let mut h: Vec<F> = (0..layer_size)
            .map(|t| self.circuit.h(layer_start + t))
            .collect();

        // Fold with r_{j-1}, r_{j-2}, …, r_1 (most recent first).
        // Paper §5.4 fold rule:
        //   h'[k] = h[2k]   + r · h[2k+2]   for k even
        //   h'[k] = h[2k-1] + r · h[2k+1]   for k odd
        for ki in (0..challenges.len()).rev() {
            let r = challenges[ki];
            let half = h.len() / 2;
            let mut h_new = Vec::with_capacity(half);
            for k in 0..half {
                let val = if k % 2 == 0 {
                    h[2 * k] + r * h[2 * k + 2]
                } else {
                    h[2 * k - 1] + r * h[2 * k + 1]
                };
                h_new.push(val);
            }
            h = h_new;
        }

        // Canonical: s_j(X) = h[0] + h[1]·X
        RoundPoly::new(h[0], h[1])
    }

    /// Produce the full [`SumcheckProof`] given all `n` challenges.
    ///
    /// # Panics
    /// Panics if `challenges.len() != num_vars`.
    pub fn prove(&self, challenges: &[F]) -> SumcheckProof<F> {
        let n = self.circuit.num_vars();
        assert_eq!(
            challenges.len(),
            n,
            "expected {n} challenges, got {}",
            challenges.len()
        );
        let claimed_sum = self.claimed_sum();
        let round_polys = (1..=n)
            .map(|j| self.compute_round_poly(&challenges[..j - 1]))
            .collect();
        SumcheckProof::new(claimed_sum, round_polys)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LagrangeProver
// ─────────────────────────────────────────────────────────────────────────────

/// Sumcheck prover for the **Lagrange basis**.
pub struct LagrangeProver<F: Field> {
    circuit: LagrangeSumCircuit<F>,
}

impl<F: Field> LagrangeProver<F> {
    /// Build the prover from a Lagrange polynomial. Cost: `O(N)`.
    pub fn new(f: &LagrangePoly<F>) -> Self {
        Self {
            circuit: LagrangeSumCircuit::build(f),
        }
    }

    /// The claimed sum `H(f)`.
    #[inline]
    pub fn claimed_sum(&self) -> F {
        self.circuit.root()
    }

    /// Number of variables `n`.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.circuit.num_vars()
    }

    /// Compute `s_j(X_j)` given challenges `r_1, …, r_{j-1}`.
    ///
    /// Uses the optimized Lagrange fold `u + r·(v − u)` and returns
    /// the round polynomial via `from_evaluations(h[0], h[1])` because
    /// in the Lagrange basis:
    /// ```text
    /// s_j(X) = h[0]·(1−X) + h[1]·X = h[0] + (h[1]−h[0])·X
    /// ```
    ///
    /// # Panics
    /// Panics if `challenges.len() >= num_vars`.
    pub fn compute_round_poly(&self, challenges: &[F]) -> RoundPoly<F> {
        let j = challenges.len() + 1;
        debug_assert!(j <= self.circuit.num_vars());

        let layer_start = 1usize << j;
        let layer_size = 1usize << j;
        let mut h: Vec<F> = (0..layer_size)
            .map(|t| self.circuit.h(layer_start + t))
            .collect();

        // Fold with r_{j-1}, r_{j-2}, …, r_1 (most recent first).
        // Lagrange optimized fold rule:
        //   h'[k] = h[2k]   + r · (h[2k+2] − h[2k])    for k even
        //   h'[k] = h[2k-1] + r · (h[2k+1] − h[2k-1])  for k odd
        for ki in (0..challenges.len()).rev() {
            let r = challenges[ki];
            let half = h.len() / 2;
            let mut h_new = Vec::with_capacity(half);
            for k in 0..half {
                let val = if k % 2 == 0 {
                    let u = h[2 * k];
                    let v = h[2 * k + 2];
                    u + r * (v - u)
                } else {
                    let u = h[2 * k - 1];
                    let v = h[2 * k + 1];
                    u + r * (v - u)
                };
                h_new.push(val);
            }
            h = h_new;
        }

        // Lagrange: s_j(X) = h[0]·(1−X) + h[1]·X
        // = h[0] + (h[1]−h[0])·X  →  from_evaluations(s(0), s(1))
        RoundPoly::from_evaluations(h[0], h[1])
    }

    /// Produce the full [`SumcheckProof`] given all `n` challenges.
    ///
    /// # Panics
    /// Panics if `challenges.len() != num_vars`.
    pub fn prove(&self, challenges: &[F]) -> SumcheckProof<F> {
        let n = self.circuit.num_vars();
        assert_eq!(
            challenges.len(),
            n,
            "expected {n} challenges, got {}",
            challenges.len()
        );
        let claimed_sum = self.claimed_sum();
        let round_polys = (1..=n)
            .map(|j| self.compute_round_poly(&challenges[..j - 1]))
            .collect();
        SumcheckProof::new(claimed_sum, round_polys)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::LagrangeDecomp;
    use crate::poly::MlPoly;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    fn canon(coeffs: &[u64]) -> CanonicalPoly<Fr> {
        CanonicalPoly::new(coeffs.iter().map(|&v| fr(v)).collect())
    }

    fn lagrange(evals: &[u64]) -> LagrangePoly<Fr> {
        LagrangePoly::new(evals.iter().map(|&v| fr(v)).collect())
    }

    // ── CanonicalProver ───────────────────────────────────────────────────────

    #[test]
    fn canonical_claimed_sum_equals_hypercube_sum_n2() {
        let f = canon(&[1, 2, 3, 4]);
        let p = CanonicalProver::new(&f);
        assert_eq!(p.claimed_sum(), fr(18));
        assert_eq!(p.claimed_sum(), f.hypercube_sum());
    }

    #[test]
    fn canonical_claimed_sum_equals_hypercube_sum_n3() {
        let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = CanonicalProver::new(&f);
        assert_eq!(p.claimed_sum(), f.hypercube_sum());
    }

    #[test]
    fn canonical_num_vars() {
        assert_eq!(CanonicalProver::new(&canon(&[1; 8])).num_vars(), 3);
    }

    #[test]
    fn canonical_round_1_is_h2_h3() {
        let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = CanonicalProver::new(&f);
        let sc = CanonicalSumCircuit::build(&f);
        let s1 = p.compute_round_poly(&[]);
        assert_eq!(s1.a, sc.h(2));
        assert_eq!(s1.b, sc.h(3));
    }

    #[test]
    fn canonical_round_1_sums_to_claimed() {
        let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = CanonicalProver::new(&f);
        assert_eq!(
            p.compute_round_poly(&[]).sum_over_boolean(),
            p.claimed_sum()
        );
    }

    #[test]
    fn canonical_round_1_sums_to_claimed_n2() {
        let f = canon(&[1, 2, 3, 4]);
        let p = CanonicalProver::new(&f);
        assert_eq!(p.compute_round_poly(&[]).sum_over_boolean(), fr(18));
    }

    #[test]
    fn canonical_round_consistency_n3() {
        let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = CanonicalProver::new(&f);
        let ch = [fr(3), fr(7), fr(11)];
        let s1 = p.compute_round_poly(&ch[..0]);
        let s2 = p.compute_round_poly(&ch[..1]);
        let s3 = p.compute_round_poly(&ch[..2]);
        assert_eq!(s1.sum_over_boolean(), p.claimed_sum());
        assert_eq!(s2.sum_over_boolean(), s1.eval(ch[0]));
        assert_eq!(s3.sum_over_boolean(), s2.eval(ch[1]));
    }

    #[test]
    fn canonical_round_consistency_n4() {
        let f = canon(&(1u64..=16).collect::<Vec<_>>());
        let p = CanonicalProver::new(&f);
        let ch = [fr(2), fr(5), fr(11), fr(7)];
        let mut prev = p.claimed_sum();
        for j in 1..=4 {
            let sj = p.compute_round_poly(&ch[..j - 1]);
            assert_eq!(sj.sum_over_boolean(), prev, "round {j}");
            prev = sj.eval(ch[j - 1]);
        }
    }

    #[test]
    fn canonical_prove_n3() {
        let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = CanonicalProver::new(&f);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = p.prove(&ch);
        assert_eq!(proof.num_vars(), 3);
        let mut prev = proof.claimed_sum;
        for j in 1..=3 {
            let sj = proof.round_poly(j);
            assert_eq!(sj.sum_over_boolean(), prev, "round {j}");
            prev = sj.eval(ch[j - 1]);
        }
    }

    #[test]
    fn canonical_proof_size_is_2n_plus_1() {
        let ch: Vec<Fr> = (0..3).map(|i| fr(i + 1)).collect();
        let proof = CanonicalProver::new(&canon(&[1; 8])).prove(&ch);
        assert_eq!(proof.size_in_field_elements(), 7);
    }

    #[test]
    #[should_panic]
    fn canonical_prove_wrong_challenge_count_panics() {
        CanonicalProver::new(&canon(&[1; 8])).prove(&[fr(1), fr(2)]);
    }

    #[test]
    fn canonical_zero_poly_claimed_sum_is_zero() {
        assert_eq!(
            CanonicalProver::new(&CanonicalPoly::<Fr>::zero(3)).claimed_sum(),
            Fr::zero()
        );
    }

    // ── LagrangeProver ────────────────────────────────────────────────────────

    #[test]
    fn lagrange_claimed_sum_equals_hypercube_sum_n2() {
        let f = lagrange(&[1, 3, 4, 10]);
        let p = LagrangeProver::new(&f);
        assert_eq!(p.claimed_sum(), fr(18));
        assert_eq!(p.claimed_sum(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_claimed_sum_n3() {
        let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = LagrangeProver::new(&f);
        assert_eq!(p.claimed_sum(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_round_1_sums_to_claimed() {
        let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = LagrangeProver::new(&f);
        assert_eq!(
            p.compute_round_poly(&[]).sum_over_boolean(),
            p.claimed_sum()
        );
    }

    #[test]
    fn lagrange_round_consistency_n3() {
        let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = LagrangeProver::new(&f);
        let ch = [fr(3), fr(7), fr(11)];
        let mut prev = p.claimed_sum();
        for j in 1..=3 {
            let sj = p.compute_round_poly(&ch[..j - 1]);
            assert_eq!(sj.sum_over_boolean(), prev, "round {j}");
            prev = sj.eval(ch[j - 1]);
        }
    }

    #[test]
    fn lagrange_round_consistency_n4() {
        let evals: Vec<u64> = (0..16).map(|i| i * 2 + 1).collect();
        let f = lagrange(&evals);
        let p = LagrangeProver::new(&f);
        let ch = [fr(2), fr(5), fr(11), fr(7)];
        let mut prev = p.claimed_sum();
        for j in 1..=4 {
            let sj = p.compute_round_poly(&ch[..j - 1]);
            assert_eq!(sj.sum_over_boolean(), prev, "round {j}");
            prev = sj.eval(ch[j - 1]);
        }
    }

    #[test]
    fn lagrange_prove_n3() {
        let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let p = LagrangeProver::new(&f);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = p.prove(&ch);
        assert_eq!(proof.num_vars(), 3);
        let mut prev = proof.claimed_sum;
        for j in 1..=3 {
            let sj = proof.round_poly(j);
            assert_eq!(sj.sum_over_boolean(), prev, "round {j}");
            prev = sj.eval(ch[j - 1]);
        }
    }

    #[test]
    fn lagrange_zero_poly_claimed_sum_is_zero() {
        assert_eq!(
            LagrangeProver::new(&LagrangePoly::<Fr>::zero(3)).claimed_sum(),
            Fr::zero()
        );
    }

    // ── Canonical and Lagrange agree ──────────────────────────────────────────

    #[test]
    fn canonical_and_lagrange_agree_n3() {
        let canon_f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
        let cp = CanonicalProver::new(&canon_f);
        let lp = LagrangeProver::new(&lagrange_f);
        assert_eq!(cp.claimed_sum(), lp.claimed_sum());
        let ch = [fr(3), fr(7), fr(11)];
        for j in 1..=3 {
            assert_eq!(
                cp.compute_round_poly(&ch[..j - 1]),
                lp.compute_round_poly(&ch[..j - 1]),
                "round {j}"
            );
        }
    }

    #[test]
    fn canonical_and_lagrange_agree_n4() {
        let canon_f = canon(&(1u64..=16).collect::<Vec<_>>());
        let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
        let cp = CanonicalProver::new(&canon_f);
        let lp = LagrangeProver::new(&lagrange_f);
        assert_eq!(cp.claimed_sum(), lp.claimed_sum());
        let ch = [fr(2), fr(5), fr(11), fr(7)];
        for j in 1..=4 {
            assert_eq!(
                cp.compute_round_poly(&ch[..j - 1]),
                lp.compute_round_poly(&ch[..j - 1]),
                "round {j}"
            );
        }
    }
}
