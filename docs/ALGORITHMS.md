# Algorithms

This note documents the mathematical structure represented by the code. It is intentionally implementation-oriented: notation is kept close to the Rust types and tree indices.

## 1. Multilinear representations

For `n` variables, let `N = 2^n`.

### Canonical basis

`CanonicalPoly<F>` stores `N` coefficients. Index `j` encodes the monomial by the binary expansion of `j`; a set bit selects the corresponding variable.

The direct evaluator computes every monomial independently. The circuit evaluator instead folds one variable at a time:

```text
u, v -> u + r v
```

until one field element remains.

### Lagrange basis

`LagrangePoly<F>` stores the `N` evaluations on the Boolean hypercube. Evaluation at an arbitrary point uses repeated interpolation folds.

Standard fold:

```text
(1-r)u + rv
```

Optimized fold:

```text
u + r(v-u)
```

The optimized form removes one multiplication per pair at the cost of an additional addition/subtraction.

## 2. Tree decomposition

The canonical decomposition splits a polynomial

```text
q = a + x_i b
```

into the children `a` and `b` by taking even- and odd-indexed coefficients.

The Lagrange-conversion decomposition uses the children

```text
p_{2j}   = a
p_{2j+1} = a + b
```

so that the leaf layer becomes the Boolean evaluation table, up to bit reversal.

For `N = 2^n`, the full binary tree contains `2N-1` nodes and the total number of field elements stored across all layers is `(n+1)N`.

The conversion performs exactly

```text
n * 2^(n-1)
```

field additions.

## 3. Bit-reversal order

Repeated low-variable-first splitting naturally produces a bit-reversed leaf order. The implementation caches the permutation and applies its inverse when converting the leaf layer into a standard `LagrangePoly` evaluation vector.

Bit reversal is an involution, so the same table serves both directions.

## 4. Boolean sum circuits

For each decomposition node, the code stores the Boolean hypercube sum of the sub-polynomial rooted at that node.

### Canonical recurrence

```text
h_j = 2 h_{2j} + h_{2j+1}
```

The implementation uses additions (`h_{2j} + h_{2j} + h_{2j+1}`) rather than a general field multiplication by `2`.

### Lagrange recurrence

```text
h_j = h_{2j} + h_{2j+1}
```

In both cases the root is the claimed Boolean sum

```text
H(f) = sum_{x in {0,1}^n} f(x).
```

## 5. Sumcheck

For a multilinear polynomial, each Sumcheck message is a degree-1 univariate polynomial

```text
s_j(X) = a_j + b_j X.
```

The proof stores one claimed sum and two field elements per round, giving `2n+1` field elements in total.

The verifier checks:

```text
s_1(0) + s_1(1) = claimed_sum
```

then, for later rounds,

```text
s_j(0) + s_j(1) = s_{j-1}(r_{j-1}),
```

and finally

```text
s_n(r_n) = f(r_1,...,r_n).
```

The final value is supplied as an oracle evaluation. This repository does not provide the commitment layer that would authenticate that oracle value in a complete proof system.

## 6. Canonical and Lagrange provers

Both provers precompute a scalar sum circuit and then extract/fold one tree layer per Sumcheck round.

The canonical prover uses the canonical recurrence directly. The Lagrange prover uses interpolation folds of the form

```text
u + r(v-u).
```

The integration suite checks that, after converting the same polynomial between bases, both provers agree on the claimed sum, oracle evaluation, and round polynomials.

## 7. Scope

The implementation is useful for studying:

- basis-dependent arithmetic costs;
- tree representations of multilinear evaluation;
- Sumcheck prover structure;
- proof transcript size;
- cross-basis equivalence;
- implementation-level performance.

It intentionally stops before Fiat-Shamir, polynomial commitments, zero-knowledge masking, recursion, and complete proof-system composition.
