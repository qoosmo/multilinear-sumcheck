use ark_ff::Field;

use super::bit_reverse_cache::get_or_build;
use crate::poly::{CanonicalPoly, LagrangePoly, MlPoly};

/// The full sequence `(p_j)_{1 ≤ j ≤ 2N-1}` from the canonical-to-Lagrange
/// circuit decomposition of `f`.
///
/// # Gate rule
///
/// At layer `i`, each node `p_j = a + x_i · b` produces two children:
///
/// ```text
/// p_{2j}   = a         (restriction to x_i = 0, even-indexed coefficients)
/// p_{2j+1} = a + b     (restriction to x_i = 1, requires one addition per coeff)
/// ```
///
/// # Leaves
///
/// The leaves `p_N, …, p_{2N-1}` are field elements forming the evaluation
/// table of `f` over `{0,1}^n`, in **bit-reverse permutation order**.
/// Use [`LagrangeDecomp::to_lagrange`] to get a properly ordered
/// [`LagrangePoly`].
///
/// # Cost
///
/// Exactly `n · 2^{n-1}` field additions. The implementation records
/// this count and the test suite checks it for multiple dimensions.
pub struct LagrangeDecomp<F: Field> {
    /// Number of variables `n`.
    pub n: usize,

    /// `N = 2^n`.
    pub big_n: usize,

    /// The sequence `(p_j)_{1 ≤ j ≤ 2N-1}`, stored 0-based.
    /// 1-based tree index `j` → `nodes[j - 1]`.
    pub nodes: Vec<CanonicalPoly<F>>,

    /// Cached bit-reverse permutation table of length `N`.
    pub bit_rev_table: Vec<usize>,

    /// Total field additions performed during construction.
    /// Must equal `n · 2^{n-1}`.
    pub addition_count: usize,
}

impl<F: Field> LagrangeDecomp<F> {
    /// Build the `(p_j)` decomposition from `f` in the canonical basis.
    ///
    /// # Panics
    /// Panics if `f.num_evals()` is not a power of two greater than zero.
    pub fn build(f: &CanonicalPoly<F>) -> Self {
        let n = f.num_vars();
        let big_n = f.num_evals();

        let mut nodes: Vec<Option<CanonicalPoly<F>>> = vec![None; 2 * big_n - 1];
        let mut addition_count = 0usize;

        // p_1 = f  (root)
        nodes[0] = Some(f.clone());

        // Layer i = 1, …, n in the 1-based tree notation.
        // At layer i, nodes are at 1-based indices [2^{i-1}, 2^i - 1].
        for i in 1..=n {
            let layer_start = 1usize << (i - 1); // 2^{i-1}  (1-based)
            let layer_end = 1usize << i; // 2^i      (exclusive, 1-based)

            for j in layer_start..layer_end {
                // Take p_j out of its slot.
                let pj = nodes[j - 1].take().expect("node must be initialised");
                let coeffs = pj.coeffs().to_vec();
                let half = coeffs.len() / 2;

                // Split p_j = a + x_i · b:
                //   a  = even-indexed coefficients  (p_{2j})
                //   b  = odd-indexed coefficients
                //   a+b = coefficient-wise sum       (p_{2j+1})
                let a: Vec<F> = (0..half).map(|r| coeffs[2 * r]).collect();
                let b: Vec<F> = (0..half).map(|r| coeffs[2 * r + 1]).collect();

                // p_{2j+1} = a + b  — costs `half` field additions
                let a_plus_b: Vec<F> = (0..half).map(|r| a[r] + b[r]).collect();
                addition_count += half;

                // Put p_j back.
                nodes[j - 1] = Some(pj);

                // Store children.
                nodes[2 * j - 1] = Some(CanonicalPoly::new(a));
                nodes[2 * j] = Some(CanonicalPoly::new(a_plus_b));
            }
        }

        // Unwrap all slots.
        let nodes: Vec<CanonicalPoly<F>> = nodes
            .into_iter()
            .enumerate()
            .map(|(idx, opt)| opt.unwrap_or_else(|| panic!("node p_{} was not filled", idx + 1)))
            .collect();

        let bit_rev_table = get_or_build(n).into_owned();

        Self {
            n,
            big_n,
            nodes,
            bit_rev_table,
            addition_count,
        }
    }

    // ── Accessors (paper notation) ────────────────────────────────────────────

    /// Return a reference to `p_j` using the **1-based paper index**.
    ///
    /// # Panics
    /// Panics if `j == 0` or `j > 2N - 1`.
    #[inline]
    pub fn p(&self, j: usize) -> &CanonicalPoly<F> {
        assert!(
            j >= 1 && j < 2 * self.big_n,
            "p index {j} out of range [1, {}]",
            2 * self.big_n - 1
        );
        &self.nodes[j - 1]
    }

    /// The root polynomial `p_1 = f`.
    #[inline]
    pub fn root(&self) -> &CanonicalPoly<F> {
        &self.nodes[0]
    }

    /// The leaf slice: `p_N, p_{N+1}, …, p_{2N-1}`.
    ///
    /// Each leaf is a degree-0 polynomial holding a single field element.
    /// They are in **bit-reverse permutation order**:
    /// leaf `k` (0-based) holds `f(rev(k))`.
    pub fn leaves(&self) -> &[CanonicalPoly<F>] {
        &self.nodes[self.big_n - 1..2 * self.big_n - 1]
    }

    /// Layer `i` (1-based, matching the paper): slice of nodes at depth `i`.
    ///
    /// - Layer `1` : `[p_1]`  (the root)
    /// - Layer `i` : `p_{2^{i-1}}, …, p_{2^i - 1}`
    /// - Layer `n+1` : the `N` leaves
    ///
    /// # Panics
    /// Panics if `i == 0` or `i > n + 1`.
    pub fn layer(&self, i: usize) -> &[CanonicalPoly<F>] {
        assert!(
            i >= 1 && i <= self.n + 1,
            "layer {i} out of range [1, {}]",
            self.n + 1
        );
        let start = (1usize << (i - 1)) - 1;
        let end = (1usize << i) - 1;
        &self.nodes[start..end]
    }

    // ── Conversion to LagrangePoly ────────────────────────────────────────────

    /// Convert the leaf layer to a properly ordered [`LagrangePoly<F>`].
    ///
    /// The leaves are in bit-reverse order: leaf `k` = `f(rev(k))`.
    /// We apply the inverse permutation (which equals the permutation itself,
    /// since bit-reverse is an involution) to recover the standard order
    /// `(f(0), f(1), …, f(N-1))`.
    pub fn to_lagrange(&self) -> LagrangePoly<F> {
        let leaves = self.leaves();
        let mut evals = vec![F::zero(); self.big_n];

        for (k, leaf) in leaves.iter().enumerate().take(self.big_n) {
            let rev_k = self.bit_rev_table[k];
            // leaf k holds f(rev(k)), so f(rev(k)) goes to position rev(k).
            evals[rev_k] = leaf.coeffs()[0];
        }

        LagrangePoly::new(evals)
    }

    // ── Structural properties ─────────────────────────────────────────────────

    /// Total field elements stored across all nodes: `(n+1) · N`.
    pub fn total_field_elements(&self) -> usize {
        self.nodes.iter().map(|p| p.num_evals()).sum()
    }

    /// Verify the addition count matches the theoretical value `n · 2^{n-1}`.
    pub fn addition_count_is_correct(&self) -> bool {
        self.addition_count == self.n * (self.big_n / 2)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::MlPoly;
    use ark_bn254::Fr;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    // ── Structural properties ─────────────────────────────────────────────────

    #[test]
    fn node_count_is_2n_minus_1() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert_eq!(d.nodes.len(), 15);
    }

    #[test]
    fn root_equals_input() {
        let coeffs: Vec<Fr> = (0..8).map(fr).collect();
        let f = CanonicalPoly::new(coeffs.clone());
        let d = LagrangeDecomp::build(&f);
        assert_eq!(d.root().coeffs(), f.coeffs());
    }

    #[test]
    fn total_field_elements_is_n_plus_1_times_n() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert_eq!(d.total_field_elements(), (d.n + 1) * d.big_n);
    }

    #[test]
    fn layer_sizes_are_correct() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert_eq!(d.layer(1).len(), 1);
        assert_eq!(d.layer(2).len(), 2);
        assert_eq!(d.layer(3).len(), 4);
        assert_eq!(d.layer(4).len(), 8);
    }

    #[test]
    fn node_coeff_sizes_decrease_by_layer() {
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert_eq!(d.p(1).num_evals(), 8);
        assert_eq!(d.p(2).num_evals(), 4);
        assert_eq!(d.p(3).num_evals(), 4);
        assert_eq!(d.p(4).num_evals(), 2);
        assert_eq!(d.p(8).num_evals(), 1);
    }

    // ── Gate rule: p_{2j} = a, p_{2j+1} = a + b ──────────────────────────────

    /// Paper example: f = α₀ + α₁x₁ + α₂x₂ + α₄x₃ + α₃x₁x₂ + α₅x₁x₃ + α₆x₂x₃ + α₇x₁x₂x₃
    /// coeffs = [1, 2, 3, 4, 5, 6, 7, 8]
    ///
    /// p_1 = f = a + x₁·b  where:
    ///   a = [α₀,α₂,α₄,α₆] = [1,3,5,7]   (even indices)
    ///   b = [α₁,α₃,α₅,α₇] = [2,4,6,8]   (odd indices)
    ///
    /// p_2 = a       = [1,3,5,7]
    /// p_3 = a + b   = [3,7,11,15]
    #[test]
    fn gate_rule_first_layer_n3() {
        let coeffs = vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)];
        let f = CanonicalPoly::new(coeffs);
        let d = LagrangeDecomp::build(&f);

        assert_eq!(d.p(2).coeffs(), &[fr(1), fr(3), fr(5), fr(7)]);
        assert_eq!(d.p(3).coeffs(), &[fr(3), fr(7), fr(11), fr(15)]);
    }

    #[test]
    fn gate_rule_second_layer_n3() {
        // p_2 = [1,3,5,7]:  a=[1,5], b=[3,7]
        //   p_4 = a     = [1,5]
        //   p_5 = a+b   = [4,12]
        //
        // p_3 = [3,7,11,15]: a=[3,11], b=[7,15]
        //   p_6 = a     = [3,11]
        //   p_7 = a+b   = [10,26]
        let coeffs = vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)];
        let f = CanonicalPoly::new(coeffs);
        let d = LagrangeDecomp::build(&f);

        assert_eq!(d.p(4).coeffs(), &[fr(1), fr(5)]);
        assert_eq!(d.p(5).coeffs(), &[fr(4), fr(12)]);
        assert_eq!(d.p(6).coeffs(), &[fr(3), fr(11)]);
        assert_eq!(d.p(7).coeffs(), &[fr(10), fr(26)]);
    }

    // ── Addition count matches theory ─────────────────────────────────────────

    #[test]
    fn addition_count_matches_theory_n3() {
        // n · 2^{n-1} = 3 · 4 = 12
        let f = CanonicalPoly::new((0..8).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert!(d.addition_count_is_correct());
        assert_eq!(d.addition_count, 12);
    }

    #[test]
    fn addition_count_matches_theory_n4() {
        // n · 2^{n-1} = 4 · 8 = 32
        let f = CanonicalPoly::new((0..16).map(fr).collect());
        let d = LagrangeDecomp::build(&f);
        assert!(d.addition_count_is_correct());
        assert_eq!(d.addition_count, 32);
    }

    #[test]
    fn addition_count_matches_theory_n1() {
        // n · 2^{n-1} = 1 · 1 = 1
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        let d = LagrangeDecomp::build(&f);
        assert!(d.addition_count_is_correct());
        assert_eq!(d.addition_count, 1);
    }

    // ── Leaves are correct evaluations ────────────────────────────────────────

    /// For f = 1 + 2x₁ + 3x₂ + 4x₁x₂  (n=2):
    /// f(0,0)=1, f(1,0)=3, f(0,1)=4, f(1,1)=10
    /// Leaf order (bit-reverse for n=2): rev(0)=0, rev(1)=2, rev(2)=1, rev(3)=3
    /// So leaves = [f(0), f(2), f(1), f(3)] = [1, 4, 3, 10]
    #[test]
    fn leaves_are_correct_evaluations_n2() {
        let f = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        let d = LagrangeDecomp::build(&f);
        let leaves = d.leaves();
        // Each leaf is a single-coefficient poly.
        assert_eq!(leaves[0].coeffs()[0], fr(1)); // f(0,0) = 1
        assert_eq!(leaves[1].coeffs()[0], fr(4)); // f(0,1) = 4
        assert_eq!(leaves[2].coeffs()[0], fr(3)); // f(1,0) = 3
        assert_eq!(leaves[3].coeffs()[0], fr(10)); // f(1,1) = 10
    }

    // ── to_lagrange produces correctly ordered evaluation table ───────────────

    #[test]
    fn to_lagrange_correct_n2() {
        // f = 1 + 2x₁ + 3x₂ + 4x₁x₂
        // Standard Lagrange order: [f(0,0), f(1,0), f(0,1), f(1,1)] = [1, 3, 4, 10]
        let f = CanonicalPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]);
        let d = LagrangeDecomp::build(&f);
        let lag = d.to_lagrange();
        assert_eq!(lag.evals(), &[fr(1), fr(3), fr(4), fr(10)]);
    }

    #[test]
    fn to_lagrange_hypercube_sum_matches_canonical_n3() {
        // H(f) must be the same whether computed from canonical or Lagrange.
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let f = CanonicalPoly::new(coeffs);
        let d = LagrangeDecomp::build(&f);
        let lag = d.to_lagrange();
        assert_eq!(f.hypercube_sum(), lag.hypercube_sum());
    }

    #[test]
    fn to_lagrange_evals_match_canonical_eval_circuit_n3() {
        // Every entry of the Lagrange eval vector must equal eval_circuit
        // of the canonical poly at the corresponding Boolean point.
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let f = CanonicalPoly::new(coeffs);
        let d = LagrangeDecomp::build(&f);
        let lag = d.to_lagrange();

        for b in 0..8usize {
            let point: Vec<Fr> = (0..3).map(|k| Fr::from(((b >> k) & 1) as u64)).collect();
            assert_eq!(lag.evals()[b], f.eval_circuit(&point), "mismatch at b={b}");
        }
    }

    #[test]
    fn to_lagrange_n1() {
        // f = 3 + 7x₁:  f(0)=3, f(1)=10
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        let d = LagrangeDecomp::build(&f);
        let lag = d.to_lagrange();
        assert_eq!(lag.evals(), &[fr(3), fr(10)]);
    }
}
