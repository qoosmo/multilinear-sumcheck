//! Benchmarks for `multilinear_sumcheck`.
//!
//! # Groups
//!
//! 1. `multilinear_eval`   — four evaluation kernels for `LagrangePoly`
//! 2. `sumcheck_canonical` — `CanonicalProver::prove` + `Verifier::verify`
//! 3. `sumcheck_lagrange`  — `LagrangeProver::prove`  + `Verifier::verify`
//! 4. `prover_construction` — one-time prover/circuit construction costs
//!
//! # Running
//!
//! ```bash
//! cargo bench --bench sumcheck
//! ```
//!
//! HTML report: `target/criterion/report/index.html`

use ark_bn254::Fr;
use ark_ff::UniformRand;
use ark_poly::DenseMultilinearExtension;
use ark_poly::MultilinearExtension;
use ark_std::rand::rngs::StdRng;
use ark_std::rand::SeedableRng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use multilinear_sumcheck::circuit::LagrangeDecomp;
use multilinear_sumcheck::poly::{CanonicalPoly, LagrangePoly};
use multilinear_sumcheck::sumcheck::prover::{CanonicalProver, LagrangeProver};
use multilinear_sumcheck::sumcheck::verifier::Verifier;

// ─────────────────────────────────────────────────────────────────────────────
// Input generation
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic RNG — same seed = same inputs across runs.
fn make_rng() -> StdRng {
    StdRng::seed_from_u64(0x1234_5678_abcd_ef01)
}

/// Random evaluation vector + point of length `n`.
fn make_lagrange_inputs(n: usize) -> (Vec<Fr>, Vec<Fr>) {
    let mut rng = make_rng();
    let big_n = 1usize << n;
    let evals: Vec<Fr> = (0..big_n).map(|_| Fr::rand(&mut rng)).collect();
    let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    (evals, point)
}

/// Random canonical coefficient vector + point of length `n`.
fn make_canonical_inputs(n: usize) -> (Vec<Fr>, Vec<Fr>) {
    let mut rng = make_rng();
    let big_n = 1usize << n;
    let coeffs: Vec<Fr> = (0..big_n).map(|_| Fr::rand(&mut rng)).collect();
    let point: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    (coeffs, point)
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 1: multilinear evaluation kernels
// ─────────────────────────────────────────────────────────────────────────────

fn bench_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("multilinear_eval");

    for n in [10, 15, 20] {
        let (evals, point) = make_lagrange_inputs(n);

        // 1. eval_standard
        {
            let poly = LagrangePoly::new(evals.clone());
            group.bench_with_input(BenchmarkId::new("eval_standard", n), &n, |b, _| {
                b.iter(|| poly.eval_standard(&point))
            });
        }

        // 2. eval_optimized
        {
            let poly = LagrangePoly::new(evals.clone());
            group.bench_with_input(BenchmarkId::new("eval_optimized", n), &n, |b, _| {
                b.iter(|| poly.eval_optimized(&point))
            });
        }

        // 3. eval_parallel
        {
            let poly = LagrangePoly::new(evals.clone());
            group.bench_with_input(BenchmarkId::new("eval_parallel", n), &n, |b, _| {
                b.iter(|| poly.eval_parallel(&point))
            });
        }

        // 4. ark-poly DenseMultilinearExtension::evaluate
        {
            let ark_poly = DenseMultilinearExtension::from_evaluations_vec(n, evals.clone());
            group.bench_with_input(BenchmarkId::new("ark_poly_evaluate", n), &n, |b, _| {
                b.iter(|| ark_poly.evaluate(&point))
            });
        }
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 2: Sumcheck canonical prover + verifier
// ─────────────────────────────────────────────────────────────────────────────

fn bench_sumcheck_canonical(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_canonical");

    for n in [10, 15, 20] {
        let (coeffs, challenges) = make_canonical_inputs(n);
        let f = CanonicalPoly::new(coeffs);

        // Build prover once — measures only the prove() call, not construction.
        let prover = CanonicalProver::new(&f);

        // Prover: build the full proof transcript.
        group.bench_with_input(BenchmarkId::new("prove", n), &n, |b, _| {
            b.iter(|| prover.prove(&challenges))
        });

        // Verifier: verify an already-produced proof.
        // The verifier is O(n) and should be very fast.
        {
            let proof = prover.prove(&challenges);
            let oracle_eval = f.eval_circuit(&challenges);
            group.bench_with_input(BenchmarkId::new("verify", n), &n, |b, _| {
                b.iter(|| {
                    Verifier::verify(&proof, &challenges, oracle_eval)
                        .expect("valid proof must verify")
                })
            });
        }

        // End-to-end: prove + verify in a single timed block.
        group.bench_with_input(BenchmarkId::new("prove_and_verify", n), &n, |b, _| {
            b.iter(|| {
                let proof = prover.prove(&challenges);
                let oracle_eval = f.eval_circuit(&challenges);
                Verifier::verify(&proof, &challenges, oracle_eval).expect("valid proof must verify")
            })
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 3: Sumcheck Lagrange prover + verifier
// ─────────────────────────────────────────────────────────────────────────────

fn bench_sumcheck_lagrange(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_lagrange");

    for n in [10, 15, 20] {
        let (evals, challenges) = make_lagrange_inputs(n);
        let f = LagrangePoly::new(evals);

        let prover = LagrangeProver::new(&f);

        // Prover
        group.bench_with_input(BenchmarkId::new("prove", n), &n, |b, _| {
            b.iter(|| prover.prove(&challenges))
        });

        // Verifier
        {
            let proof = prover.prove(&challenges);
            let oracle_eval = f.eval_optimized(&challenges);
            group.bench_with_input(BenchmarkId::new("verify", n), &n, |b, _| {
                b.iter(|| {
                    Verifier::verify(&proof, &challenges, oracle_eval)
                        .expect("valid proof must verify")
                })
            });
        }

        // End-to-end
        group.bench_with_input(BenchmarkId::new("prove_and_verify", n), &n, |b, _| {
            b.iter(|| {
                let proof = prover.prove(&challenges);
                let oracle_eval = f.eval_optimized(&challenges);
                Verifier::verify(&proof, &challenges, oracle_eval).expect("valid proof must verify")
            })
        });
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Group 4: Prover construction cost
// ─────────────────────────────────────────────────────────────────────────────
//
// Measures the one-time cost of building the sum circuit.
// This is paid once per polynomial and amortized across all prove() calls.

fn bench_prover_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prover_construction");

    for n in [10, 15, 20] {
        let (coeffs, _) = make_canonical_inputs(n);
        let (evals, _) = make_lagrange_inputs(n);

        let canon_f = CanonicalPoly::new(coeffs);
        let lagrange_f = LagrangePoly::new(evals);

        group.bench_with_input(BenchmarkId::new("canonical_new", n), &n, |b, _| {
            b.iter(|| CanonicalProver::new(&canon_f))
        });

        group.bench_with_input(BenchmarkId::new("lagrange_new", n), &n, |b, _| {
            b.iter(|| LagrangeProver::new(&lagrange_f))
        });

        // Also measure LagrangeDecomp::build + to_lagrange conversion cost
        // since that is the typical pipeline starting from a canonical poly.
        group.bench_with_input(
            BenchmarkId::new("canonical_to_lagrange_conversion", n),
            &n,
            |b, _| b.iter(|| LagrangeDecomp::build(&canon_f).to_lagrange()),
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Criterion entry point
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_eval,
    bench_sumcheck_canonical,
    bench_sumcheck_lagrange,
    bench_prover_construction,
);
criterion_main!(benches);
