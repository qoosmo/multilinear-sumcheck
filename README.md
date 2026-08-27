# Multilinear Sumcheck

[![CI](https://github.com/qoosmo/multilinear-sumcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/qoosmo/multilinear-sumcheck/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

**Basis-aware multilinear polynomial algorithms and a research implementation of the Sumcheck protocol in Rust.**

This repository studies how the representation of a multilinear polynomial changes the structure and cost of evaluation and Sumcheck. It implements the same multilinear object in both the canonical/monomial basis and the Lagrange/evaluation basis over the Boolean hypercube, then exposes the tree decompositions and scalar sum circuits that connect those representations to Sumcheck.

The goal is not to provide a production SNARK. The goal is to make the algebraic structure explicit, testable, benchmarkable, and easy to inspect.

## What is implemented

- **Canonical multilinear polynomials** stored as dense monomial coefficients.
- **Lagrange multilinear polynomials** stored as evaluations over `{0,1}^n`.
- **Linear-time canonical evaluation** by repeated variable folding.
- **Standard, optimized, and Rayon-parallel Lagrange evaluation kernels**.
- **Canonical tree decomposition** by even/odd coefficient splitting.
- **Canonical → Lagrange conversion** through a tree whose leaves are Boolean-hypercube evaluations in bit-reversed order.
- **Bit-reversal caching** used by the decomposition and sum circuits.
- **Boolean sum circuits** for both bases.
- **Canonical and Lagrange Sumcheck provers**.
- **Stateless Sumcheck verifier** with explicit round-consistency and final-oracle checks.
- **End-to-end integration tests** covering completeness, transcript tampering, cross-basis agreement, edge cases, and zero-variable transcripts.
- **Criterion benchmarks** for evaluation, proving, verification, and prover construction.

## Why the basis matters

Let `f : F^n -> F` be multilinear and let `N = 2^n`.

In the canonical basis we store

```text
f(x_1,...,x_n) = sum_{S subseteq [n]} alpha_S prod_{i in S} x_i.
```

In the Lagrange basis we store the evaluation table

```text
(f(0), f(1), ..., f(N-1))
```

over the Boolean hypercube.

Both contain the same mathematical information, but they induce different local recurrences and different arithmetic trade-offs. This repository makes those differences explicit rather than hiding them behind a generic polynomial interface.

## Core architecture

```text
Canonical coefficients
        |
        | even/odd tree decomposition
        v
  canonical q_j tree
        |
        +--------------------------+
        |                          |
        v                          v
bit-reversed leaves          canonical sum circuit h_j
        |                          |
        | a / (a+b) gates          | Sumcheck round extraction
        v                          v
Lagrange evaluations        Canonical Sumcheck prover
        |                          |
        v                          |
Lagrange sum circuit h_j           |
        |                          |
        v                          v
Lagrange Sumcheck prover ----> SumcheckProof
                                  |
                                  v
                               Verifier
                                  |
                                  v
                         final oracle evaluation
```

## Selected complexity properties

For `N = 2^n`:

| Operation | Implementation | Arithmetic / asymptotic cost |
| --- | --- | --- |
| Canonical evaluation | `CanonicalPoly::eval_naive` | `O(N log N)` |
| Canonical evaluation | `CanonicalPoly::eval_circuit` | `N-1` multiplications + `N-1` additions |
| Lagrange evaluation | `eval_standard` | fold `(1-r)u + rv` |
| Lagrange evaluation | `eval_optimized` | fold `u + r(v-u)`, one multiplication per pair |
| Canonical → Lagrange decomposition | `LagrangeDecomp::build` | `n * 2^(n-1)` additions |
| Sum-circuit construction | canonical / Lagrange | `O(N)` |
| Sumcheck proof size | `SumcheckProof` | `2n + 1` field elements |
| Verifier | `Verifier` | `O(n)` round checks + final oracle check |

The benchmark suite measures implementation-level runtime separately from these field-operation counts.

## Quick start

Requirements:

- stable Rust toolchain
- Cargo

```bash
cargo test --all-features
cargo run --example basic_sumcheck
cargo bench --bench sumcheck
```

## Minimal example

```rust
use ark_bn254::Fr;
use multilinear_sumcheck::poly::CanonicalPoly;
use multilinear_sumcheck::sumcheck::{CanonicalProver, Verifier};

fn main() {
    let f = CanonicalPoly::new(
        (1u64..=8).map(Fr::from).collect()
    );
    let challenges = [Fr::from(3u64), Fr::from(7u64), Fr::from(11u64)];

    let prover = CanonicalProver::new(&f);
    let proof = prover.prove(&challenges);
    let oracle_eval = f.eval_circuit(&challenges);

    Verifier::verify(&proof, &challenges, oracle_eval)
        .expect("valid Sumcheck proof");
}
```

A complete runnable version is available in [`examples/basic_sumcheck.rs`](examples/basic_sumcheck.rs).

## Sumcheck boundary

This repository implements the **algebraic Sumcheck core**. In particular:

1. the prover constructs one degree-1 round polynomial per variable;
2. the verifier checks Boolean sums across rounds;
3. the verifier checks the last round against an externally supplied evaluation `f(r_1,...,r_n)`.

The current code intentionally does **not** implement:

- Fiat-Shamir transcript generation;
- a polynomial commitment scheme;
- Merkle commitments;
- zero-knowledge masking/blinding;
- recursive composition;
- a complete SNARK or STARK;
- production hardening or side-channel guarantees.

This separation is deliberate: it keeps the repository focused on the multilinear/Sumcheck algebra and makes the trust boundary explicit.

## Testing

The integration suite checks:

- honest canonical proofs verify;
- honest Lagrange proofs verify;
- modified claimed sums are rejected;
- modified intermediate round polynomials are rejected;
- incorrect oracle values are rejected;
- canonical and Lagrange representations agree on the same polynomial;
- proof size scales as `2n+1` field elements;
- zero and constant polynomials behave correctly;
- zero-variable transcripts are handled without panics.

Run:

```bash
cargo test --all-features
```

## Benchmarks

Criterion benchmarks cover:

- standard vs optimized vs parallel multilinear evaluation;
- comparison with `ark-poly::DenseMultilinearExtension`;
- canonical Sumcheck proving and verification;
- Lagrange Sumcheck proving and verification;
- prover construction;
- canonical → Lagrange conversion.

The benchmark inputs use deterministic randomness and currently include `n = 10, 15, 20`.

```bash
cargo bench --bench sumcheck
```

Criterion writes its HTML report to:

```text
target/criterion/report/index.html
```

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for reproducibility guidance and a results template. No performance claims are published without the machine/toolchain context needed to reproduce them.

## Repository layout

```text
src/
  poly/       canonical and Lagrange multilinear representations
  circuit/    decomposition trees, bit reversal, and Boolean sum circuits
  sumcheck/   proof transcript, provers, and verifier
benches/      Criterion benchmark suite
tests/        end-to-end integration tests
examples/     minimal runnable examples
docs/         algorithm and benchmark notes
```

## Research status

This is a research-oriented implementation intended for experimentation, validation of algebraic recurrences, operation-count accounting, and benchmarking. APIs may evolve as the underlying research evolves.

For a more detailed walk-through of the algorithms, see [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Security

This repository is **not production cryptography**. Do not use it to secure funds or sensitive systems without a separate security review and the missing protocol layers described above.

See [`SECURITY.md`](SECURITY.md) for reporting guidance.

## Authors

- Ali Mkhida
- Adil Iguider

## License

Licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)); or
- MIT License ([`LICENSE-MIT`](LICENSE-MIT)).

at your option.
