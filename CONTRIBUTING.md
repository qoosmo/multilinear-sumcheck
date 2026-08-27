# Contributing

Contributions should preserve the repository's research-oriented goals: explicit algebra, reproducible tests, and measurable performance.

Before opening a pull request, run:

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo bench --bench sumcheck --no-run
```

For algorithmic changes, please include:

- the mathematical recurrence or invariant being implemented;
- the expected asymptotic and/or field-operation cost;
- correctness tests, ideally including a cross-check against another representation;
- benchmark evidence when performance is part of the claim.

Avoid performance claims that are not accompanied by enough machine/toolchain information to reproduce them.
