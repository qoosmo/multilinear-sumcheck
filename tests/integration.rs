//! End-to-end integration tests for the `multilinear_sumcheck` Sumcheck protocol.
//!
//! These tests cross all module boundaries and verify the full pipeline:
//!
//! ```text
//! CanonicalPoly / LagrangePoly
//!      ↓  (circuit decomposition)
//! CanonicalSumCircuit / LagrangeSumCircuit
//!      ↓  (prover)
//! SumcheckProof
//!      ↓  (verifier + oracle)
//! Ok(()) / Err(VerifierError)
//! ```
//!
//! # Test categories
//!
//! 1. **Completeness** — honest proofs always verify.
//! 2. **Soundness**    — tampered proofs are always rejected.
//! 3. **Cross-basis**  — canonical and Lagrange proofs agree.
//! 4. **Edge cases**   — zero polynomial, `n=1`, large `n`.

use ark_bn254::Fr;
use ark_ff::{UniformRand, Zero};
use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;

use multilinear_sumcheck::circuit::LagrangeDecomp;
use multilinear_sumcheck::poly::{CanonicalPoly, LagrangePoly, MlPoly};
use multilinear_sumcheck::sumcheck::proof::RoundPoly;
use multilinear_sumcheck::sumcheck::prover::{CanonicalProver, LagrangeProver};
use multilinear_sumcheck::sumcheck::verifier::{Verifier, VerifierError};

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

fn fr(n: u64) -> Fr {
    Fr::from(n)
}

/// Deterministic RNG — same seed across all tests.
fn rng() -> StdRng {
    StdRng::seed_from_u64(0xdead_beef_cafe_1234)
}

/// Random canonical polynomial in `n` variables.
fn random_canonical(n: usize) -> CanonicalPoly<Fr> {
    let mut rng = rng();
    let coeffs: Vec<Fr> = (0..1usize << n).map(|_| Fr::rand(&mut rng)).collect();
    CanonicalPoly::new(coeffs)
}

/// Random Lagrange polynomial in `n` variables.
fn random_lagrange(n: usize) -> LagrangePoly<Fr> {
    let mut rng = rng();
    let evals: Vec<Fr> = (0..1usize << n).map(|_| Fr::rand(&mut rng)).collect();
    LagrangePoly::new(evals)
}

/// Random challenge vector of length `n`.
fn random_challenges(n: usize) -> Vec<Fr> {
    let mut rng = rng();
    (0..n).map(|_| Fr::rand(&mut rng)).collect()
}

/// Small canonical polynomial from a u64 slice.
fn canon(coeffs: &[u64]) -> CanonicalPoly<Fr> {
    CanonicalPoly::new(coeffs.iter().map(|&v| fr(v)).collect())
}

/// Small Lagrange polynomial from a u64 slice.
fn lagrange(evals: &[u64]) -> LagrangePoly<Fr> {
    LagrangePoly::new(evals.iter().map(|&v| fr(v)).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Completeness — canonical prover
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn canonical_completeness_n1() {
    // f = 3 + 7x₁  →  H(f) = 3 + 3+7 = 13... wait: H = f(0)+f(1) = 3+10 = 13
    // canonical: H = 3·2^1 + 7·2^0 = 6+7 = 13
    let f = canon(&[3, 7]);
    let ch = vec![fr(5)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn canonical_completeness_n2() {
    let f = canon(&[1, 2, 3, 4]);
    let ch = vec![fr(5), fr(9)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn canonical_completeness_n3() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn canonical_completeness_n4() {
    let f = canon(&(1u64..=16).collect::<Vec<_>>());
    let ch = vec![fr(2), fr(5), fr(11), fr(7)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn canonical_completeness_random_n5() {
    let f = random_canonical(5);
    let ch = random_challenges(5);
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn canonical_completeness_random_n10() {
    // Uses the cached bit-reverse table for n=10.
    let f = random_canonical(10);
    let ch = random_challenges(10);
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Completeness — Lagrange prover
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lagrange_completeness_n1() {
    // f(0)=3, f(1)=10  →  H = 13
    let f = lagrange(&[3, 10]);
    let ch = vec![fr(5)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn lagrange_completeness_n2() {
    let f = lagrange(&[1, 3, 4, 10]);
    let ch = vec![fr(5), fr(9)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn lagrange_completeness_n3() {
    let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn lagrange_completeness_n4() {
    let evals: Vec<u64> = (0..16).map(|i| i * 2 + 1).collect();
    let f = lagrange(&evals);
    let ch = vec![fr(2), fr(5), fr(11), fr(7)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn lagrange_completeness_random_n5() {
    let f = random_lagrange(5);
    let ch = random_challenges(5);
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn lagrange_completeness_random_n10() {
    let f = random_lagrange(10);
    let ch = random_challenges(10);
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Soundness — tampered proofs are rejected
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn soundness_tampered_claimed_sum_canonical() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let mut proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);

    proof.claimed_sum = proof.claimed_sum + fr(1);

    assert!(matches!(
        Verifier::verify(&proof, &ch, oracle),
        Err(VerifierError::Round1SumMismatch { .. })
    ));
}

#[test]
fn soundness_tampered_claimed_sum_lagrange() {
    let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let mut proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);

    proof.claimed_sum = fr(0);

    assert!(matches!(
        Verifier::verify(&proof, &ch, oracle),
        Err(VerifierError::Round1SumMismatch { .. })
    ));
}

#[test]
fn soundness_tampered_mid_round_poly_canonical() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let mut proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);

    // Replace round 2 with a random polynomial.
    proof.round_polys[1] = RoundPoly::new(fr(999), fr(888));

    assert!(matches!(
        Verifier::verify(&proof, &ch, oracle),
        Err(VerifierError::RoundConsistencyMismatch { round: 2, .. })
    ));
}

#[test]
fn soundness_tampered_mid_round_poly_lagrange() {
    let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let mut proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);

    proof.round_polys[1] = RoundPoly::new(fr(777), fr(666));

    assert!(matches!(
        Verifier::verify(&proof, &ch, oracle),
        Err(VerifierError::RoundConsistencyMismatch { round: 2, .. })
    ));
}

#[test]
fn soundness_wrong_oracle_canonical() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = CanonicalProver::new(&f).prove(&ch);

    // Correct proof but wrong oracle evaluation.
    assert!(matches!(
        Verifier::verify(&proof, &ch, fr(999)),
        Err(VerifierError::OracleCheckFailed { .. })
    ));
}

#[test]
fn soundness_wrong_oracle_lagrange() {
    let f = lagrange(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = LagrangeProver::new(&f).prove(&ch);

    assert!(matches!(
        Verifier::verify(&proof, &ch, fr(0)),
        Err(VerifierError::OracleCheckFailed { .. })
    ));
}

#[test]
fn soundness_wrong_challenge_count() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = CanonicalProver::new(&f).prove(&ch);

    // Pass only 2 challenges for a 3-variable proof.
    assert!(matches!(
        Verifier::verify(&proof, &[fr(3), fr(7)], fr(0)),
        Err(VerifierError::ChallengeLengthMismatch {
            expected: 3,
            got: 2
        })
    ));
}

#[test]
fn soundness_completely_random_proof_rejected() {
    // Build a valid proof structure with random garbage values.
    let mut rng = rng();
    let n = 4;
    let round_polys: Vec<RoundPoly<Fr>> = (0..n)
        .map(|_| RoundPoly::new(Fr::rand(&mut rng), Fr::rand(&mut rng)))
        .collect();
    let proof =
        multilinear_sumcheck::sumcheck::proof::SumcheckProof::new(Fr::rand(&mut rng), round_polys);
    let ch = random_challenges(n);
    let oracle = Fr::rand(&mut rng);

    // A random proof almost surely fails — check it is indeed rejected.
    assert!(Verifier::verify(&proof, &ch, oracle).is_err());
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Cross-basis — canonical and Lagrange provers agree
// ─────────────────────────────────────────────────────────────────────────────

/// For a polynomial expressed in both bases, the two provers must:
/// - agree on the claimed sum
/// - produce round polynomials that evaluate identically
/// - both produce proofs that verify against the same oracle
#[test]
fn cross_basis_agree_n3() {
    let canon_f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
    let ch = vec![fr(3), fr(7), fr(11)];

    let cp = CanonicalProver::new(&canon_f);
    let lp = LagrangeProver::new(&lagrange_f);

    // Same claimed sum.
    assert_eq!(cp.claimed_sum(), lp.claimed_sum());

    // Same round polynomials.
    for j in 1..=3 {
        assert_eq!(
            cp.compute_round_poly(&ch[..j - 1]),
            lp.compute_round_poly(&ch[..j - 1]),
            "round {j}"
        );
    }

    // Both oracle evaluations agree.
    let oracle_c = canon_f.eval_circuit(&ch);
    let oracle_l = lagrange_f.eval_optimized(&ch);
    assert_eq!(oracle_c, oracle_l);

    // Both proofs verify.
    assert!(Verifier::verify(&cp.prove(&ch), &ch, oracle_c).is_ok());
    assert!(Verifier::verify(&lp.prove(&ch), &ch, oracle_l).is_ok());
}

#[test]
fn cross_basis_agree_n4() {
    let canon_f = canon(&(1u64..=16).collect::<Vec<_>>());
    let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
    let ch = vec![fr(2), fr(5), fr(11), fr(7)];

    let oracle_c = canon_f.eval_circuit(&ch);
    let oracle_l = lagrange_f.eval_optimized(&ch);
    assert_eq!(oracle_c, oracle_l);

    assert!(Verifier::verify(&CanonicalProver::new(&canon_f).prove(&ch), &ch, oracle_c).is_ok());
    assert!(Verifier::verify(&LagrangeProver::new(&lagrange_f).prove(&ch), &ch, oracle_l).is_ok());
}

#[test]
fn cross_basis_agree_random_n5() {
    let canon_f = random_canonical(5);
    let lagrange_f = LagrangeDecomp::build(&canon_f).to_lagrange();
    let ch = random_challenges(5);

    let oracle_c = canon_f.eval_circuit(&ch);
    let oracle_l = lagrange_f.eval_optimized(&ch);
    assert_eq!(oracle_c, oracle_l);

    assert!(Verifier::verify(&CanonicalProver::new(&canon_f).prove(&ch), &ch, oracle_c).is_ok());
    assert!(Verifier::verify(&LagrangeProver::new(&lagrange_f).prove(&ch), &ch, oracle_l).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Edge cases
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn zero_polynomial_canonical() {
    let f = CanonicalPoly::<Fr>::zero(3);
    let ch = vec![fr(1), fr(2), fr(3)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);

    assert_eq!(proof.claimed_sum, Fr::zero());
    assert_eq!(oracle, Fr::zero());
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn zero_polynomial_lagrange() {
    let f = LagrangePoly::<Fr>::zero(3);
    let ch = vec![fr(1), fr(2), fr(3)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    let oracle = f.eval_optimized(&ch);

    assert_eq!(proof.claimed_sum, Fr::zero());
    assert_eq!(oracle, Fr::zero());
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn constant_polynomial_canonical() {
    // f = 5 (constant), H(f) = 5 * 2^n = 5 * 8 = 40 for n=3
    let mut coeffs = vec![Fr::zero(); 8];
    coeffs[0] = fr(5);
    let f = CanonicalPoly::new(coeffs);
    let ch = vec![fr(2), fr(3), fr(7)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    let oracle = f.eval_circuit(&ch);

    assert_eq!(proof.claimed_sum, fr(40));
    assert_eq!(oracle, fr(5)); // f(r) = 5 for any r
    assert!(Verifier::verify(&proof, &ch, oracle).is_ok());
}

#[test]
fn proof_size_grows_linearly_with_n() {
    // Proof has 2n+1 field elements.
    for n in [1, 2, 3, 4, 5] {
        let f = random_canonical(n);
        let ch = random_challenges(n);
        let proof = CanonicalProver::new(&f).prove(&ch);
        assert_eq!(
            proof.size_in_field_elements(),
            2 * n + 1,
            "proof size wrong for n={n}"
        );
    }
}

#[test]
fn claimed_sum_matches_hypercube_sum_canonical() {
    for n in [1, 2, 3, 4] {
        let f = random_canonical(n);
        let proof = CanonicalProver::new(&f).prove(&random_challenges(n));
        assert_eq!(
            proof.claimed_sum,
            f.hypercube_sum(),
            "claimed sum mismatch for n={n}"
        );
    }
}

#[test]
fn claimed_sum_matches_hypercube_sum_lagrange() {
    for n in [1, 2, 3, 4] {
        let f = random_lagrange(n);
        let proof = LagrangeProver::new(&f).prove(&random_challenges(n));
        assert_eq!(
            proof.claimed_sum,
            f.hypercube_sum(),
            "claimed sum mismatch for n={n}"
        );
    }
}

#[test]
fn verify_transcript_canonical_n3() {
    let f = canon(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let ch = vec![fr(3), fr(7), fr(11)];
    let proof = CanonicalProver::new(&f).prove(&ch);
    // Internal consistency check without oracle.
    assert!(Verifier::verify_transcript(&proof, &ch).is_ok());
}

#[test]
fn verify_transcript_lagrange_n4() {
    let evals: Vec<u64> = (0..16).map(|i| i * 3 + 1).collect();
    let f = lagrange(&evals);
    let ch = vec![fr(2), fr(5), fr(9), fr(13)];
    let proof = LagrangeProver::new(&f).prove(&ch);
    assert!(Verifier::verify_transcript(&proof, &ch).is_ok());
}
