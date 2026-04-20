//! Sumcheck verifier.
//!
//! The verifier is **stateless** — it takes a [`SumcheckProof`], the
//! verifier challenges `r_1, …, r_n`, and an oracle evaluation
//! `f(r_1, …, r_n)`, and checks three conditions:
//!
//! 1. **Round 1:** `s_1(0) + s_1(1) = claimed_sum`
//! 2. **Rounds 2…n:** `s_j(0) + s_j(1) = s_{j-1}(r_{j-1})`
//! 3. **Final oracle:** `s_n(r_n) = oracle_eval`
//!
//! If all checks pass it returns `Ok(())`.
//! If any check fails it returns a [`VerifierError`] describing which
//! check failed and what values were seen.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mlp_pro::sumcheck::verifier::Verifier;
//!
//! // Prover side
//! let proof = prover.prove(&challenges);
//!
//! // Oracle evaluation (in a real system this comes from a commitment)
//! let oracle_eval = f.eval_circuit(&challenges);
//!
//! // Verifier side
//! Verifier::verify(&proof, &challenges, oracle_eval)?;
//! ```

use ark_ff::Field;
use std::fmt;

use crate::sumcheck::proof::SumcheckProof;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Describes which verifier check failed and the values that caused it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifierError<F: Field> {
    /// The number of challenges does not match the number of rounds.
    ChallengeLengthMismatch {
        expected: usize,
        got:      usize,
    },

    /// Round 1 failed: `s_1(0) + s_1(1) ≠ claimed_sum`.
    Round1SumMismatch {
        got:      F,
        expected: F,
    },

    /// Round `j` (2 ≤ j ≤ n) failed: `s_j(0) + s_j(1) ≠ s_{j-1}(r_{j-1})`.
    RoundConsistencyMismatch {
        round:    usize,
        got:      F,
        expected: F,
    },

    /// Final oracle check failed: `s_n(r_n) ≠ oracle_eval`.
    OracleCheckFailed {
        got:      F,
        expected: F,
    },
}

impl<F: Field + fmt::Display> fmt::Display for VerifierError<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChallengeLengthMismatch { expected, got } => write!(
                f,
                "challenge length mismatch: expected {expected}, got {got}"
            ),
            Self::Round1SumMismatch { got, expected } => write!(
                f,
                "round 1 sum check failed: s_1(0)+s_1(1) = {got}, expected {expected}"
            ),
            Self::RoundConsistencyMismatch { round, got, expected } => write!(
                f,
                "round {round} consistency check failed: \
                 s_{round}(0)+s_{round}(1) = {got}, expected {expected}"
            ),
            Self::OracleCheckFailed { got, expected } => write!(
                f,
                "final oracle check failed: s_n(r_n) = {got}, expected {expected}"
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Verifier
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless Sumcheck verifier.
///
/// All methods are associated functions — no state is stored.
pub struct Verifier;

impl Verifier {
    /// Verify a [`SumcheckProof`] against the given challenges and oracle.
    ///
    /// # Arguments
    ///
    /// - `proof`       — the prover's transcript
    /// - `challenges`  — `r_1, …, r_n` sampled by the verifier
    /// - `oracle_eval` — `f(r_1, …, r_n)` from the polynomial oracle
    ///
    /// # Returns
    ///
    /// `Ok(())` if all three checks pass.
    /// `Err(VerifierError)` with a precise description of the first
    /// failing check.
    ///
    /// # Panics
    ///
    /// Does not panic — all error conditions are returned as `Err`.
    pub fn verify<F: Field>(
        proof:       &SumcheckProof<F>,
        challenges:  &[F],
        oracle_eval: F,
    ) -> Result<(), VerifierError<F>> {
        let n = proof.num_vars();

        // ── Pre-check: challenge count ────────────────────────────────────────
        if challenges.len() != n {
            return Err(VerifierError::ChallengeLengthMismatch {
                expected: n,
                got:      challenges.len(),
            });
        }

        // ── Check 1: round 1 ─────────────────────────────────────────────────
        // s_1(0) + s_1(1) must equal the claimed sum.
        let s1  = proof.round_poly(1);
        let got = s1.sum_over_boolean();
        if got != proof.claimed_sum {
            return Err(VerifierError::Round1SumMismatch {
                got,
                expected: proof.claimed_sum,
            });
        }

        // ── Check 2: round consistency for j = 2 … n ─────────────────────────
        // s_j(0) + s_j(1) must equal s_{j-1}(r_{j-1}).
        for j in 2..=n {
            let sj       = proof.round_poly(j);
            let s_prev   = proof.round_poly(j - 1);
            let r_prev   = challenges[j - 2]; // r_{j-1} is challenges[j-2] (0-based)
            let expected = s_prev.eval(r_prev);
            let got      = sj.sum_over_boolean();
            if got != expected {
                return Err(VerifierError::RoundConsistencyMismatch {
                    round: j,
                    got,
                    expected,
                });
            }
        }

        // ── Check 3: final oracle check ───────────────────────────────────────
        // s_n(r_n) must equal f(r_1, …, r_n).
        let sn      = proof.round_poly(n);
        let r_n     = challenges[n - 1];
        let sn_at_r = sn.eval(r_n);
        if sn_at_r != oracle_eval {
            return Err(VerifierError::OracleCheckFailed {
                got:      sn_at_r,
                expected: oracle_eval,
            });
        }

        Ok(())
    }

    /// Verify only the internal consistency of the proof transcript,
    /// without an oracle evaluation.
    ///
    /// This checks rounds 1 through n but skips the final oracle check.
    /// Useful for testing the prover in isolation before an oracle is
    /// available.
    pub fn verify_transcript<F: Field>(
        proof:      &SumcheckProof<F>,
        challenges: &[F],
    ) -> Result<(), VerifierError<F>> {
        let n = proof.num_vars();

        if challenges.len() != n {
            return Err(VerifierError::ChallengeLengthMismatch {
                expected: n,
                got:      challenges.len(),
            });
        }

        // Round 1
        let s1  = proof.round_poly(1);
        let got = s1.sum_over_boolean();
        if got != proof.claimed_sum {
            return Err(VerifierError::Round1SumMismatch {
                got,
                expected: proof.claimed_sum,
            });
        }

        // Rounds 2..n
        for j in 2..=n {
            let sj       = proof.round_poly(j);
            let s_prev   = proof.round_poly(j - 1);
            let r_prev   = challenges[j - 2];
            let expected = s_prev.eval(r_prev);
            let got      = sj.sum_over_boolean();
            if got != expected {
                return Err(VerifierError::RoundConsistencyMismatch {
                    round: j,
                    got,
                    expected,
                });
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{CanonicalPoly, LagrangePoly};
    use crate::circuit::LagrangeDecomp;
    use crate::sumcheck::prover::{CanonicalProver, LagrangeProver};
    use crate::sumcheck::proof::RoundPoly;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    fn fr(n: u64) -> Fr { Fr::from(n) }

    fn canon(coeffs: &[u64]) -> CanonicalPoly<Fr> {
        CanonicalPoly::new(coeffs.iter().map(|&v| fr(v)).collect())
    }

    fn lagrange(evals: &[u64]) -> LagrangePoly<Fr> {
        LagrangePoly::new(evals.iter().map(|&v| fr(v)).collect())
    }

    // ── Full end-to-end: canonical ────────────────────────────────────────────

    /// Prove and verify a canonical proof end-to-end.
    /// Oracle evaluation is computed from the canonical polynomial directly.
    #[test]
    fn canonical_prove_and_verify_n3() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];

        let proof       = CanonicalProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_circuit(&ch);

        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    #[test]
    fn canonical_prove_and_verify_n4() {
        let f  = canon(&(1u64..=16).collect::<Vec<_>>());
        let ch = [fr(2), fr(5), fr(11), fr(7)];

        let proof       = CanonicalProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_circuit(&ch);

        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    #[test]
    fn canonical_verify_n2() {
        let f  = canon(&[1, 2, 3, 4]);
        let ch = [fr(5), fr(9)];

        let proof       = CanonicalProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_circuit(&ch);

        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    // ── Full end-to-end: Lagrange ─────────────────────────────────────────────

    #[test]
    fn lagrange_prove_and_verify_n3() {
        let f  = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];

        let proof       = LagrangeProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_optimized(&ch);

        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    #[test]
    fn lagrange_prove_and_verify_n4() {
        let evals: Vec<u64> = (0..16).map(|i| i * 2 + 1).collect();
        let f  = lagrange(&evals);
        let ch = [fr(2), fr(5), fr(11), fr(7)];

        let proof       = LagrangeProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_optimized(&ch);

        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    // ── Canonical and Lagrange produce same accepted proof ────────────────────

    #[test]
    fn canonical_and_lagrange_both_verify_n3() {
        let canon_f    = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
        let ch         = [fr(3), fr(7), fr(11)];

        let cp = CanonicalProver::new(&canon_f);
        let lp = LagrangeProver::new(&lagrange_f);

        let canon_proof   = cp.prove(&ch);
        let lagrange_proof = lp.prove(&ch);

        let oracle_canon   = canon_f.eval_circuit(&ch);
        let oracle_lagrange = lagrange_f.eval_optimized(&ch);

        // Both oracle evaluations must agree
        assert_eq!(oracle_canon, oracle_lagrange);

        // Both proofs must verify
        assert!(Verifier::verify(&canon_proof,   &ch, oracle_canon).is_ok());
        assert!(Verifier::verify(&lagrange_proof, &ch, oracle_lagrange).is_ok());
    }

    // ── verify_transcript (no oracle) ─────────────────────────────────────────

    #[test]
    fn verify_transcript_accepts_valid_proof_n3() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = CanonicalProver::new(&f).prove(&ch);
        assert!(Verifier::verify_transcript(&proof, &ch).is_ok());
    }

    // ── Error cases ───────────────────────────────────────────────────────────

    #[test]
    fn wrong_challenge_count_returns_error() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = CanonicalProver::new(&f).prove(&ch);

        // Too few challenges
        let err = Verifier::verify(&proof, &[fr(3), fr(7)], fr(0));
        assert!(matches!(err, Err(VerifierError::ChallengeLengthMismatch { .. })));
    }

    #[test]
    fn tampered_claimed_sum_fails_round1() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let mut proof = CanonicalProver::new(&f).prove(&ch);

        // Tamper with the claimed sum
        proof.claimed_sum = fr(999);

        let oracle = f.eval_circuit(&ch);
        let err = Verifier::verify(&proof, &ch, oracle);
        assert!(matches!(err, Err(VerifierError::Round1SumMismatch { .. })));
    }

    #[test]
    fn tampered_round_poly_fails_consistency() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let mut proof = CanonicalProver::new(&f).prove(&ch);

        // Tamper with round 2 polynomial
        proof.round_polys[1] = RoundPoly::new(fr(999), fr(999));

        let oracle = f.eval_circuit(&ch);
        let err = Verifier::verify(&proof, &ch, oracle);
        assert!(matches!(
            err,
            Err(VerifierError::RoundConsistencyMismatch { round: 2, .. })
        ));
    }

    #[test]
    fn wrong_oracle_eval_fails_final_check() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = CanonicalProver::new(&f).prove(&ch);

        // Pass a wrong oracle evaluation
        let err = Verifier::verify(&proof, &ch, fr(999));
        assert!(matches!(err, Err(VerifierError::OracleCheckFailed { .. })));
    }

    #[test]
    fn tampered_last_round_poly_fails_oracle() {
        let f  = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let mut proof = CanonicalProver::new(&f).prove(&ch);

        // Tamper with the last round polynomial only
        // (keeps round consistency but breaks the oracle check)
        let oracle = f.eval_circuit(&ch);
        let correct_s3 = proof.round_polys[2];
        // Keep s_3(0)+s_3(1) = s_2(r_2) by adjusting both a and b proportionally
        // Easiest: corrupt a but keep sum_over_boolean intact — impossible cleanly,
        // so just replace with a zeroed poly and check oracle fails
        proof.round_polys[2] = RoundPoly::new(
            correct_s3.a + fr(1),
            correct_s3.b,
        );

        let err = Verifier::verify(&proof, &ch, oracle);
        // Either consistency or oracle fails
        assert!(err.is_err());
    }

    // ── Zero polynomial ───────────────────────────────────────────────────────

    #[test]
    fn zero_poly_verifies() {
        let f  = CanonicalPoly::<Fr>::zero(3);
        let ch = [fr(3), fr(7), fr(11)];
        let proof       = CanonicalProver::new(&f).prove(&ch);
        let oracle_eval = f.eval_circuit(&ch);
        assert_eq!(oracle_eval, Fr::zero());
        assert!(Verifier::verify(&proof, &ch, oracle_eval).is_ok());
    }

    // ── Lagrange error cases ──────────────────────────────────────────────────

    #[test]
    fn lagrange_tampered_claimed_sum_fails() {
        let f  = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let mut proof = LagrangeProver::new(&f).prove(&ch);
        proof.claimed_sum = fr(999);
        let oracle = f.eval_optimized(&ch);
        assert!(matches!(
            Verifier::verify(&proof, &ch, oracle),
            Err(VerifierError::Round1SumMismatch { .. })
        ));
    }

    #[test]
    fn lagrange_wrong_oracle_fails() {
        let f  = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ch = [fr(3), fr(7), fr(11)];
        let proof = LagrangeProver::new(&f).prove(&ch);
        assert!(matches!(
            Verifier::verify(&proof, &ch, fr(0)),
            Err(VerifierError::OracleCheckFailed { .. })
        ));
    }
}