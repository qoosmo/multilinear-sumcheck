//! Boolean sum circuit — the `(h_j)` scalar tree.
//!
//! Given a multilinear polynomial `f`, the Boolean sum circuit stores
//! the sequence of field elements `(h_j)_{1 ≤ j ≤ 2N-1}` where each
//! `h_j` equals the Boolean hypercube sum of the sub-polynomial `q_j`
//! rooted at node `j` of the decomposition circuit.
//!
//! # Two variants
//!
//! | Basis     | Recurrence              | Leaf values            |
//! |-----------|-------------------------|------------------------|
//! | Canonical | `h_j = 2·h_{2j} + h_{2j+1}` | bit-rev of coefficients |
//! | Lagrange  | `h_j = h_{2j} + h_{2j+1}`   | bit-rev of evaluations  |
//!
//! In both cases `h_1 = H(f) = Σ_{x ∈ {0,1}^n} f(x)`.
//!
//! # Indexing convention
//!
//! 1-based throughout, matching the paper exactly.
//! Internal storage is 0-based: paper index `j` → `data[j-1]`.
//!
//! ```text
//! layer 0 :  j = 1                    (root  = H(f))
//! layer 1 :  j = 2, 3
//! layer i :  j ∈ [2^i, 2^{i+1} − 1]
//! layer n :  j ∈ [N, 2N−1]            (leaves = scalar coefficients / evals)
//! ```
//!
//! # Construction cost
//!
//! Building either circuit from its leaf values costs exactly
//! `N − 1` field operations (additions or fused multiply-add),
//! where `N = 2^n`.
//!
//! # Usage in the Sumcheck prover
//!
//! The Sumcheck prover reads `h_j` values layer by layer to construct
//! the round polynomials `s_k(X_k)` without re-evaluating `f`.

use ark_ff::Field;

use crate::circuit::bit_reverse_cache::get_or_build;
use crate::poly::{CanonicalPoly, LagrangePoly, MlPoly};

// ─────────────────────────────────────────────────────────────────────────────
// Shared trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for a Boolean sum circuit.
///
/// Implemented by both [`CanonicalSumCircuit`] and [`LagrangeSumCircuit`].
pub trait SumCircuit<F: Field> {
    /// Number of variables `n`.
    fn num_vars(&self) -> usize;

    /// `N = 2^n`.
    fn n(&self) -> usize {
        1 << self.num_vars()
    }

    /// Get `h_j` using the **1-based paper index**.
    ///
    /// # Panics
    /// Panics if `j == 0` or `j > 2N − 1`.
    fn h(&self, j: usize) -> F;

    /// The root value `h_1 = H(f) = Σ_{x ∈ {0,1}^n} f(x)`.
    fn root(&self) -> F {
        self.h(1)
    }

    /// The leaf slice at 0-based positions within the leaf layer.
    /// `leaves()[k]` = `h_{N+k}` = `h_{2^n + k}`.
    fn leaves(&self) -> &[F];

    /// Layer `i` (0-based): the slice `h_{2^i}, …, h_{2^{i+1}−1}`.
    ///
    /// - Layer `0`: `[h_1]`      (root)
    /// - Layer `n`: `[h_N … h_{2N-1}]` (leaves)
    fn layer(&self, i: usize) -> &[F];

    /// Verify the recurrence relation for every internal node.
    /// Used in tests — not called in production.
    fn verify_recurrence(&self) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal builder
// ─────────────────────────────────────────────────────────────────────────────

/// Build the flat `(h_j)` array bottom-up from a leaf slice.
///
/// `recurrence(left, right)` encodes the rule:
/// - canonical: `|left, right| left + left + right`
/// - Lagrange:  `|left, right| left + right`
///
/// The leaf values must already be in bit-reversed order matching the
/// `(q_j)` decomposition tree.
fn build_from_leaves<F: Field, R>(leaves: &[F], recurrence: R) -> Vec<F>
where
    R: Fn(F, F) -> F,
{
    let big_n = leaves.len();
    debug_assert!(big_n.is_power_of_two() && big_n > 0);
    let n = big_n.trailing_zeros() as usize;

    // Allocate 2N-1 slots (0-based).  data[j-1] = h_j.
    let mut data = vec![F::zero(); 2 * big_n - 1];

    // Fill leaf layer: h_N … h_{2N-1}  → data[N-1 .. 2N-2]
    for (k, &v) in leaves.iter().enumerate() {
        data[big_n - 1 + k] = v;
    }

    // Propagate bottom-up from layer n-1 down to layer 0.
    for i in (0..n).rev() {
        let layer_start_0based = (1usize << i) - 1;
        let layer_size = 1usize << i;
        for t in 0..layer_size {
            let parent = layer_start_0based + t;
            let left = 2 * parent + 1;
            let right = 2 * parent + 2;
            data[parent] = recurrence(data[left], data[right]);
        }
    }

    data
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical sum circuit
// ─────────────────────────────────────────────────────────────────────────────

/// Boolean sum circuit for a multilinear polynomial in the **canonical basis**.
///
/// Stores the scalar sequence `(h_j)_{1 ≤ j ≤ 2N-1}` where:
///
/// - **Recurrence:** `h_j = h_{2j} + h_{2j} + h_{2j+1}` for `j ∈ [1, N-1]`
///   (equivalent to `2·h_{2j} + h_{2j+1}` but using two additions
///   instead of one multiplication)
/// - **Leaves:** `h_{N+k} = α_{rev(k)}` — the bit-reversed coefficient vector
/// - **Root:** `h_1 = H(f) = Σ_{x ∈ {0,1}^n} f(x)`
///
/// # Derivation of the recurrence
///
/// The canonical gate relation is `q_j = q_{2j} + x_i · q_{2j+1}`.
/// Applying the sum operator `H_i` and using linearity:
/// ```text
/// H_i(q_j) = H_{i+1}(q_{2j}) + H_{i+1}(q_{2j}) + H_{i+1}(q_{2j+1})
///           = h_{2j} + h_{2j} + h_{2j+1}
/// ```
#[derive(Debug, Clone)]
pub struct CanonicalSumCircuit<F: Field> {
    /// `n`
    num_vars: usize,
    /// `N = 2^n`
    big_n: usize,
    /// Flat storage of length `2N − 1`.  `data[j-1]` = `h_j`.
    data: Vec<F>,
}

impl<F: Field> CanonicalSumCircuit<F> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from a [`CanonicalPoly`].
    ///
    /// Applies the bit-reverse permutation to the coefficient vector to
    /// obtain the leaf layer, then propagates bottom-up using
    /// `h_j = h_{2j} + h_{2j} + h_{2j+1}`.
    ///
    /// # Complexity
    /// `O(N)` — exactly `N − 1` pairs of additions.
    pub fn build(f: &CanonicalPoly<F>) -> Self {
        let n = f.num_vars();
        let big_n = f.num_evals();

        // For n ∈ {10, 15, 20}: zero-cost borrow of the pre-computed static table.
        // For other n: single heap allocation, used here and dropped immediately.
        let table = get_or_build(n);
        let coeffs = f.coeffs();
        let leaves: Vec<F> = (0..big_n).map(|k| coeffs[table[k]]).collect();

        // h_j = h_{2j} + h_{2j} + h_{2j+1}
        // Replaces 2·h_{2j} + h_{2j+1}: one multiplication + one addition
        // with two additions — cheaper on BN254 where mul >> add.
        let data = build_from_leaves(&leaves, |l, r| l + l + r);

        Self {
            num_vars: n,
            big_n,
            data,
        }
    }

    /// Build directly from a leaf slice already in bit-reversed order.
    ///
    /// # Panics
    /// Panics if `leaves.len()` is not a power of two or is zero.
    pub fn from_leaves(leaves: &[F]) -> Self {
        assert!(
            !leaves.is_empty() && leaves.len().is_power_of_two(),
            "CanonicalSumCircuit::from_leaves: length must be a power of two"
        );
        let big_n = leaves.len();
        let num_vars = big_n.trailing_zeros() as usize;
        let data = build_from_leaves(leaves, |l, r| l + l + r);
        Self {
            num_vars,
            big_n,
            data,
        }
    }
}

impl<F: Field> SumCircuit<F> for CanonicalSumCircuit<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn h(&self, j: usize) -> F {
        assert!(
            j >= 1 && j < 2 * self.big_n,
            "h index {j} out of range [1, {}]",
            2 * self.big_n - 1
        );
        self.data[j - 1]
    }

    fn leaves(&self) -> &[F] {
        &self.data[self.big_n - 1..2 * self.big_n - 1]
    }

    fn layer(&self, i: usize) -> &[F] {
        assert!(
            i <= self.num_vars,
            "layer {i} out of range [0, {}]",
            self.num_vars
        );
        let start = (1usize << i) - 1;
        let end = (1usize << (i + 1)) - 1;
        &self.data[start..end]
    }

    fn verify_recurrence(&self) -> bool {
        (1..self.big_n).all(|j| {
            let l = self.data[2 * j - 1];
            self.data[j - 1] == l + l + self.data[2 * j]
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lagrange sum circuit
// ─────────────────────────────────────────────────────────────────────────────

/// Boolean sum circuit for a multilinear polynomial in the **Lagrange basis**.
///
/// Stores the scalar sequence `(h_j)_{1 ≤ j ≤ 2N-1}` where:
///
/// - **Recurrence:** `h_j = h_{2j} + h_{2j+1}` for `j ∈ [1, N-1]`
/// - **Leaves:** `h_{N+k} = f(rev(k))` — the bit-reversed evaluation vector
/// - **Root:** `h_1 = H(f) = Σ_{j=0}^{N-1} f(j)`
///
/// # Derivation of the recurrence
///
/// The Lagrange gate relation is `q_j = (1−x_i)·q_{2j} + x_i·q_{2j+1}`.
/// Applying `H_i`:
/// ```text
/// H_i(q_j) = H_{i+1}(q_{2j}) + H_{i+1}(q_{2j+1})
///           = h_{2j} + h_{2j+1}
/// ```
#[derive(Debug, Clone)]
pub struct LagrangeSumCircuit<F: Field> {
    /// `n`
    num_vars: usize,
    /// `N = 2^n`
    big_n: usize,
    /// Flat storage of length `2N − 1`.  `data[j-1]` = `h_j`.
    data: Vec<F>,
}

impl<F: Field> LagrangeSumCircuit<F> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Build from a [`LagrangePoly`].
    ///
    /// Applies the bit-reverse permutation to the evaluation vector to
    /// obtain the leaf layer, then propagates bottom-up using
    /// `h_j = h_{2j} + h_{2j+1}`.
    ///
    /// # Complexity
    /// `O(N)` — exactly `N − 1` additions.
    pub fn build(f: &LagrangePoly<F>) -> Self {
        let n = f.num_vars();
        let big_n = f.num_evals();

        // For n ∈ {10, 15, 20}: zero-cost borrow of the pre-computed static table.
        // For other n: single heap allocation, used here and dropped immediately.
        let table = get_or_build(n);
        let evals = f.evals();
        let leaves: Vec<F> = (0..big_n).map(|k| evals[table[k]]).collect();

        let data = build_from_leaves(&leaves, |l, r| l + r);
        Self {
            num_vars: n,
            big_n,
            data,
        }
    }

    /// Build directly from a leaf slice already in bit-reversed order.
    ///
    /// # Panics
    /// Panics if `leaves.len()` is not a power of two or is zero.
    pub fn from_leaves(leaves: &[F]) -> Self {
        assert!(
            !leaves.is_empty() && leaves.len().is_power_of_two(),
            "LagrangeSumCircuit::from_leaves: length must be a power of two"
        );
        let big_n = leaves.len();
        let num_vars = big_n.trailing_zeros() as usize;
        let data = build_from_leaves(leaves, |l, r| l + r);
        Self {
            num_vars,
            big_n,
            data,
        }
    }
}

impl<F: Field> SumCircuit<F> for LagrangeSumCircuit<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn h(&self, j: usize) -> F {
        assert!(
            j >= 1 && j < 2 * self.big_n,
            "h index {j} out of range [1, {}]",
            2 * self.big_n - 1
        );
        self.data[j - 1]
    }

    fn leaves(&self) -> &[F] {
        &self.data[self.big_n - 1..2 * self.big_n - 1]
    }

    fn layer(&self, i: usize) -> &[F] {
        assert!(
            i <= self.num_vars,
            "layer {i} out of range [0, {}]",
            self.num_vars
        );
        let start = (1usize << i) - 1;
        let end = (1usize << (i + 1)) - 1;
        &self.data[start..end]
    }

    fn verify_recurrence(&self) -> bool {
        (1..self.big_n).all(|j| self.data[j - 1] == self.data[2 * j - 1] + self.data[2 * j])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{CanonicalPoly, LagrangePoly, MlPoly};
    use ark_bn254::Fr;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    #[test]
    fn canonical_paper_example_n3() {
        let leaves = vec![fr(1), fr(4), fr(3), fr(7), fr(2), fr(6), fr(5), fr(8)];
        let sc = CanonicalSumCircuit::from_leaves(&leaves);
        assert_eq!(sc.root(), fr(88));
        assert_eq!(sc.h(2), fr(25));
        assert_eq!(sc.h(3), fr(38));
        assert_eq!(sc.h(4), fr(6));
        assert_eq!(sc.h(5), fr(13));
        assert_eq!(sc.h(6), fr(10));
        assert_eq!(sc.h(7), fr(18));
    }

    #[test]
    fn canonical_recurrence_holds_paper_example() {
        let leaves = vec![fr(1), fr(4), fr(3), fr(7), fr(2), fr(6), fr(5), fr(8)];
        let sc = CanonicalSumCircuit::from_leaves(&leaves);
        assert!(sc.verify_recurrence());
    }

    #[test]
    fn canonical_root_equals_hypercube_sum_n2() {
        let f = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn canonical_root_equals_hypercube_sum_n3() {
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let f = CanonicalPoly::new(coeffs);
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn canonical_root_equals_hypercube_sum_n4() {
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 3 + 1)).collect();
        let f = CanonicalPoly::new(coeffs);
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn canonical_node_count() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.data.len(), 15);
    }

    #[test]
    fn canonical_layer_sizes() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.layer(0).len(), 1);
        assert_eq!(sc.layer(1).len(), 2);
        assert_eq!(sc.layer(2).len(), 4);
        assert_eq!(sc.layer(3).len(), 8);
    }

    #[test]
    fn canonical_leaves_slice_length() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.leaves().len(), 8);
    }

    #[test]
    fn canonical_recurrence_holds_n4() {
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 + 1)).collect();
        let f = CanonicalPoly::new(coeffs);
        let sc = CanonicalSumCircuit::build(&f);
        assert!(sc.verify_recurrence());
    }

    #[test]
    fn canonical_n1_edge_case() {
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        let sc = CanonicalSumCircuit::build(&f);
        assert_eq!(sc.root(), fr(13));
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_root_equals_hypercube_sum_n2() {
        let f = LagrangePoly::new(vec![fr(1), fr(3), fr(4), fr(10)]);
        let sc = LagrangeSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_root_equals_hypercube_sum_n3() {
        let evals: Vec<Fr> = (1..=8).map(fr).collect();
        let f = LagrangePoly::new(evals);
        let sc = LagrangeSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_root_equals_hypercube_sum_n4() {
        let evals: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 2 + 3)).collect();
        let f = LagrangePoly::new(evals);
        let sc = LagrangeSumCircuit::build(&f);
        assert_eq!(sc.root(), f.hypercube_sum());
    }

    #[test]
    fn lagrange_recurrence_holds_n3() {
        let evals: Vec<Fr> = (1..=8).map(fr).collect();
        let f = LagrangePoly::new(evals);
        let sc = LagrangeSumCircuit::build(&f);
        assert!(sc.verify_recurrence());
    }

    #[test]
    fn lagrange_layer_sizes() {
        let f = LagrangePoly::new((0..8).map(fr).collect());
        let sc = LagrangeSumCircuit::build(&f);
        assert_eq!(sc.layer(0).len(), 1);
        assert_eq!(sc.layer(1).len(), 2);
        assert_eq!(sc.layer(2).len(), 4);
        assert_eq!(sc.layer(3).len(), 8);
    }

    #[test]
    fn lagrange_n1_edge_case() {
        let f = LagrangePoly::new(vec![fr(5), fr(9)]);
        let sc = LagrangeSumCircuit::build(&f);
        assert_eq!(sc.root(), fr(14));
    }

    #[test]
    fn canonical_and_lagrange_agree_on_root_n3() {
        use crate::circuit::LagrangeDecomp;
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let canon = CanonicalPoly::new(coeffs);
        let lag = LagrangeDecomp::build(&canon).to_lagrange();
        let sc_canon = CanonicalSumCircuit::build(&canon);
        let sc_lag = LagrangeSumCircuit::build(&lag);
        assert_eq!(sc_canon.root(), sc_lag.root());
        assert_eq!(sc_canon.root(), canon.hypercube_sum());
    }

    #[test]
    fn canonical_and_lagrange_agree_on_root_n4() {
        use crate::circuit::LagrangeDecomp;
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 3 + 1)).collect();
        let canon = CanonicalPoly::new(coeffs);
        let lag = LagrangeDecomp::build(&canon).to_lagrange();
        let sc_canon = CanonicalSumCircuit::build(&canon);
        let sc_lag = LagrangeSumCircuit::build(&lag);
        assert_eq!(sc_canon.root(), sc_lag.root());
    }

    #[test]
    fn canonical_from_leaves_matches_build() {
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let f = CanonicalPoly::new(coeffs);
        let table = get_or_build(3);
        let manual_leaves: Vec<Fr> = (0..8).map(|k| f.coeffs()[table[k]]).collect();
        let sc_build = CanonicalSumCircuit::build(&f);
        let sc_from_leaves = CanonicalSumCircuit::from_leaves(&manual_leaves);
        assert_eq!(sc_build.data, sc_from_leaves.data);
    }

    #[test]
    fn lagrange_from_leaves_matches_build() {
        let evals: Vec<Fr> = (1..=8).map(fr).collect();
        let f = LagrangePoly::new(evals);
        let table = get_or_build(3);
        let manual_leaves: Vec<Fr> = (0..8).map(|k| f.evals()[table[k]]).collect();
        let sc_build = LagrangeSumCircuit::build(&f);
        let sc_from_leaves = LagrangeSumCircuit::from_leaves(&manual_leaves);
        assert_eq!(sc_build.data, sc_from_leaves.data);
    }
}
