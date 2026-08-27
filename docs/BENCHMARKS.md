# Benchmarking

The benchmark suite uses Criterion and deterministic random inputs so repeated runs compare the same logical instances.

## Run

```bash
cargo bench --bench sumcheck
```

The HTML report is written to:

```text
target/criterion/report/index.html
```

## Benchmark groups

1. `multilinear_eval`
   - `eval_standard`
   - `eval_optimized`
   - `eval_parallel`
   - `ark_poly_evaluate`
2. `sumcheck_canonical`
   - `prove`
   - `verify`
   - `prove_and_verify`
3. `sumcheck_lagrange`
   - `prove`
   - `verify`
   - `prove_and_verify`
4. `prover_construction`
   - canonical prover construction
   - Lagrange prover construction
   - canonical → Lagrange conversion

The current benchmark sizes are `n = 10, 15, 20`.

## Reproducibility record

Every published benchmark should include:

```text
Date:
Git commit:
OS:
CPU:
Logical cores:
RAM:
rustc --version:
cargo --version:
Build mode: release / Criterion
```

## Results template

Do not fill this table from memory or from a different machine. Record the actual Criterion estimates from the machine described above.

| n | ark-poly eval | optimized eval | parallel eval | canonical prove | Lagrange prove | verify |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 |  |  |  |  |  |  |
| 15 |  |  |  |  |  |  |
| 20 |  |  |  |  |  |  |

## Interpretation

Runtime results should be kept separate from algebraic operation counts. A method with fewer field multiplications is not automatically faster on every machine: allocation, memory bandwidth, cache behavior, parallel scheduling, and field implementation all matter.

For that reason the README states exact operation counts only when they follow directly from the algorithm, and leaves machine-dependent performance claims to reproducible benchmark output.
