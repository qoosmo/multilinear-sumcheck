use ark_ff::Field;
use rayon::prelude::*;

use crate::circuit::bit_reverse_cache::get_or_build;

// ─────────────────────────────────────────────────────────────────────────────
// UniTerm
// ─────────────────────────────────────────────────────────────────────────────

/// A single term `α · Xᵏ` in a univariate polynomial.
///
/// # Examples
///
/// For `f(X) = 3 + 5X² + 7X⁵`:
/// - `UniTerm { coeff: 3, degree: 0 }` → `3`
/// - `UniTerm { coeff: 5, degree: 2 }` → `5X²`
/// - `UniTerm { coeff: 7, degree: 5 }` → `7X⁵`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniTerm<F: Field> {
    /// The coefficient `α ∈ 𝔽`.
    pub coeff: F,
    /// The exponent `k ∈ ℕ`.
    pub degree: usize,
}

impl<F: Field> UniTerm<F> {
    /// Construct the term `coeff · Xᵈᵉᵍʳᵉᵉ`.
    pub fn new(coeff: F, degree: usize) -> Self {
        Self { coeff, degree }
    }

    /// `true` if the coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.coeff.is_zero()
    }

    /// Evaluate the term at `r`: returns `α · rᵏ`.
    ///
    /// Uses square-and-multiply — cost: `O(log k)` multiplications.
    pub fn eval(&self, r: F) -> F {
        if self.is_zero() {
            return F::zero();
        }
        self.coeff * field_pow(r, self.degree)
    }

    /// Evaluate the monomial `Xᵏ` (without the coefficient) at `r`.
    pub fn eval_monomial(&self, r: F) -> F {
        field_pow(r, self.degree)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniPoly
// ─────────────────────────────────────────────────────────────────────────────

/// A dense univariate polynomial stored as a zero-padded coefficient vector.
///
/// ```text
/// f(X) = α₀ + α₁·X + α₂·X² + … + α_{N-1}·X^{N-1}
/// ```
///
/// where `N = 2^n` is the smallest power of two ≥ the number of input
/// coefficients. Zero-padding above the actual degree does not change
/// the polynomial: `f(r)` is identical for any `r`.
///
/// # Padding rule
///
/// Given an input vector of length `m`:
/// - if `m` is already a power of two, no padding is added.
/// - otherwise the vector is padded with zeros to length `2^⌈log₂(m)⌉`.
///
/// # Relationship to [`UniTerm`]
///
/// A `UniPoly` is the sum of up to `N` terms.
/// `terms()` gives the sparse view over non-zero coefficients only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniPoly<F: Field> {
    /// Dense coefficient vector, length `N = 2^n`.
    /// `coeffs[k]` = coefficient of `Xᵏ`.
    /// Entries above `actual_degree` are always zero.
    coeffs: Vec<F>,

    /// True degree — index of the last non-zero coefficient in the
    /// original input, before padding. Stored so `degree()` is O(1).
    actual_degree: usize,
}

impl<F: Field> UniPoly<F> {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Construct from a coefficient vector of **any length ≥ 1**.
    ///
    /// Pads with zeros to the next power of two internally.
    ///
    /// # Panics
    /// Panics if `coeffs` is empty.
    pub fn new(coeffs: Vec<F>) -> Self {
        assert!(
            !coeffs.is_empty(),
            "UniPoly: coefficient vector must not be empty"
        );

        let actual_degree = coeffs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| !c.is_zero())
            .map(|(k, _)| k)
            .unwrap_or(0);

        let padded_len = coeffs.len().next_power_of_two();
        let mut padded = coeffs;
        padded.resize(padded_len, F::zero());

        Self {
            coeffs: padded,
            actual_degree,
        }
    }

    /// The zero polynomial, internally stored with `2^n` coefficients.
    pub fn zero(n: usize) -> Self {
        Self {
            coeffs: vec![F::zero(); 1 << n],
            actual_degree: 0,
        }
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// `N = 2^n` — the padded length of the coefficient vector.
    pub fn padded_len(&self) -> usize {
        self.coeffs.len()
    }

    /// `n` such that `padded_len() = 2^n`.
    pub fn log_len(&self) -> usize {
        self.coeffs.len().trailing_zeros() as usize
    }

    /// The true degree (ignoring padding zeros).
    /// Returns `0` for the zero polynomial.
    pub fn degree(&self) -> usize {
        self.actual_degree
    }

    /// `true` if the polynomial is identically zero.
    pub fn is_zero_poly(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// The full padded coefficient slice (length = `padded_len()`).
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// The coefficient of `Xᵏ`.
    /// Returns `F::zero()` for `k >= padded_len()`.
    pub fn coeff(&self, k: usize) -> F {
        if k < self.coeffs.len() {
            self.coeffs[k]
        } else {
            F::zero()
        }
    }

    // ── Term views ────────────────────────────────────────────────────────────

    /// Iterate over all non-zero terms (padding zeros excluded).
    pub fn terms(&self) -> impl Iterator<Item = UniTerm<F>> + '_ {
        self.coeffs
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(k, &coeff)| UniTerm::new(coeff, k))
    }

    /// Iterate over all terms including padding zeros.
    pub fn all_terms(&self) -> impl Iterator<Item = UniTerm<F>> + '_ {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(k, &coeff)| UniTerm::new(coeff, k))
    }

    // ── Evaluation ────────────────────────────────────────────────────────────

    /// Naive evaluation: `f(r) = Σ αₖ · rᵏ`, each term computed independently.
    ///
    /// Cost: `O(N log N)`. Correctness reference only.
    pub fn eval_naive(&self, r: F) -> F {
        self.coeffs
            .iter()
            .enumerate()
            .fold(F::zero(), |acc, (k, &alpha)| {
                if alpha.is_zero() {
                    acc
                } else {
                    acc + UniTerm::new(alpha, k).eval(r)
                }
            })
    }

    /// Horner evaluation: `f(r) = α₀ + r(α₁ + r(α₂ + … + r·α_{N-1})…)`
    ///
    /// Cost: `N-1` multiplications, `N-1` additions.
    /// Fully sequential — no parallelism possible.
    /// This is the standard baseline we compare against.
    pub fn eval_horner(&self, r: F) -> F {
        self.coeffs
            .iter()
            .rev()
            .fold(F::zero(), |acc, &alpha| acc * r + alpha)
    }

    /// Estrin evaluation: divide-and-conquer on the natural coefficient order.
    ///
    /// # Algorithm
    ///
    /// Groups adjacent pairs `(α_{2t}, α_{2t+1})` at the first layer,
    /// then combines with `r²`, then `r⁴`, and so on.
    ///
    /// ```text
    /// Layer 0 (bottom):  buf[t] = α_{2t} + r · α_{2t+1}
    /// Layer 1:           buf[t] = buf_{2t} + r² · buf_{2t+1}
    /// Layer k:           buf[t] = buf_{2t} + r^{2^k} · buf_{2t+1}
    /// ```
    ///
    /// # Cost
    /// `N-1` multiplications + `N-1` additions + `n-1` squarings.
    /// Parallel depth: `O(log N)`.
    ///
    /// Works directly on the natural coefficient order — no permutation needed.
    pub fn eval_estrin(&self, r: F) -> F {
        let n = self.log_len();

        // Power table: powers[k] = r^{2^k} for k = 0, …, n-1.
        // powers[0] = r, powers[1] = r², powers[2] = r⁴, …
        let powers = power_table(r, n);

        let mut buf = self.coeffs.clone();

        // At layer k (0-based from bottom), combine with r^{2^k}.
        for &lambda in powers.iter().take(n) {
            let new_len = buf.len() / 2;
            for t in 0..new_len {
                // buf[t] = buf[2t] + r^{2^k} · buf[2t+1]
                buf[t] = buf[2 * t] + lambda * buf[2 * t + 1];
            }
            buf.truncate(new_len);
        }

        buf[0]
    }

    /// Circuit-based sequential evaluation.
    ///
    /// # Algorithm
    ///
    /// 1. Reorder the coefficients into the bit-reversed leaf order of the
    ///    `(q_j)` circuit decomposition — using the cached table for
    ///    `n ∈ {10, 15, 20}`, or a runtime-computed table otherwise.
    /// 2. Fold bottom-up: at layer `i` from the bottom, combine pairs
    ///    with weight `r^{2^{n-1-i}}`:
    ///    ```text
    ///    buf[t] = buf[2t] + r^{2^{n-1-i}} · buf[2t+1]
    ///    ```
    ///    The weight decreases each layer: `r^{2^{n-1}}, r^{2^{n-2}}, …, r`.
    ///
    /// # Cost
    /// `N-1` multiplications + `N-1` additions + `n-1` squarings.
    /// Parallel depth: `O(log N)`.
    ///
    /// # Advantage over Estrin
    /// When the same polynomial is evaluated many times (e.g. across
    /// Sumcheck rounds), the bit-reversed layout can be cached.
    /// Each subsequent evaluation then skips the permutation step entirely.
    pub fn eval_circuit(&self, r: F) -> F {
        let n = self.log_len();
        let big_n = self.padded_len();

        // Power table: powers[k] = r^{2^k} for k = 0, …, n-1.
        let powers = power_table(r, n);

        // Build bit-reversed leaf array.
        // For n ∈ {10, 15, 20}: zero-cost borrow of the static cached table.
        // For other n: single heap allocation, used here and dropped.
        let table = get_or_build(n);
        let mut buf: Vec<F> = (0..big_n).map(|k| self.coeffs[table[k]]).collect();

        // Bottom-up fold.
        // Layer i from the bottom (i = 0 is the leaf layer):
        //   weight = powers[n-1-i] = r^{2^{n-1-i}}
        for i in 0..n {
            let lambda = powers[n - 1 - i];
            let new_len = buf.len() / 2;
            for t in 0..new_len {
                buf[t] = buf[2 * t] + lambda * buf[2 * t + 1];
            }
            buf.truncate(new_len);
        }

        buf[0]
    }

    /// Circuit-based parallel evaluation using `rayon`.
    ///
    /// Identical algorithm to [`eval_circuit`] but the inner fold loop
    /// is parallelised with `rayon::par_chunks(2)`.
    ///
    /// # Why this parallelises cleanly
    ///
    /// At each layer, every pair `(buf[2t], buf[2t+1])` is independent
    /// of every other pair, and all pairs share a **single scalar weight**
    /// `r^{2^{n-1-i}}` that is fixed for the entire layer.
    /// There is no per-element scalar read — the multiplier is broadcast.
    /// This makes the univariate parallel kernel slightly cleaner than
    /// the multilinear case (where each layer used a different `r_k`).
    ///
    /// # When this is faster
    ///
    /// Thread overhead dominates at small `n`. The parallel version is
    /// faster than `eval_circuit` only for `n ≥ 18` on most machines.
    ///
    /// # Cost
    /// Same arithmetic as `eval_circuit` — `N-1` muls + `N-1` adds +
    /// `n-1` squarings — but with `O(log N)` parallel depth.
    pub fn eval_circuit_parallel(&self, r: F) -> F
    where
        F: Send + Sync,
    {
        let n = self.log_len();
        let big_n = self.padded_len();

        let powers = power_table(r, n);

        // For n ∈ {10, 15, 20}: zero-cost borrow of the static cached table.
        let table = get_or_build(n);
        let mut buf: Vec<F> = (0..big_n).map(|k| self.coeffs[table[k]]).collect();

        let mut tmp = Vec::with_capacity(big_n / 2);

        for i in 0..n {
            let lambda = powers[n - 1 - i];
            tmp.clear();
            buf.par_chunks(2)
                .map(|pair| pair[0] + lambda * pair[1])
                .collect_into_vec(&mut tmp);
            std::mem::swap(&mut buf, &mut tmp);
        }

        buf[0]
    }

    // ── Circuit decomposition ─────────────────────────────────────────────────

    /// Full circuit decomposition producing `(q_j)_{1 ≤ j ≤ 2N-1}`.
    ///
    /// See [`UniDecomp`] for the complete documentation.
    pub fn decompose(&self) -> UniDecomp<F> {
        let n = self.log_len();
        let big_n = self.padded_len();

        let mut nodes: Vec<Vec<F>> = vec![Vec::new(); 2 * big_n - 1];
        nodes[0] = self.coeffs.clone();

        for i in 0..n {
            let layer_start = 1usize << i;
            let layer_end = 1usize << (i + 1);
            for j in layer_start..layer_end {
                let parent = nodes[j - 1].clone();
                let half = parent.len() / 2;
                let left: Vec<F> = (0..half).map(|r| parent[2 * r]).collect();
                let right: Vec<F> = (0..half).map(|r| parent[2 * r + 1]).collect();
                nodes[2 * j - 1] = left;
                nodes[2 * j] = right;
            }
        }

        UniDecomp { n, big_n, nodes }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniDecomp
// ─────────────────────────────────────────────────────────────────────────────

/// The full sequence `(q_j)_{1 ≤ j ≤ 2N-1}` from the univariate
/// circuit decomposition of `f`.
///
/// Each node is stored as a `Vec<F>` of compressed coefficients.
/// Node `q_j` at layer `i` represents a polynomial in the variable
/// `Y = X^{2^i}` — it has `N / 2^i` coefficients.
pub struct UniDecomp<F: Field> {
    /// `n` such that `N = 2^n`.
    pub n: usize,
    /// `N = 2^n`.
    pub big_n: usize,
    /// Flat storage. Paper index `j` (1-based) → `nodes[j-1]`.
    pub nodes: Vec<Vec<F>>,
}

impl<F: Field> UniDecomp<F> {
    /// `q_j` using the **1-based paper index**.
    ///
    /// # Panics
    /// Panics if `j == 0` or `j > 2N - 1`.
    #[inline]
    pub fn q(&self, j: usize) -> &[F] {
        assert!(
            j >= 1 && j < 2 * self.big_n,
            "q index {j} out of range [1, {}]",
            2 * self.big_n - 1
        );
        &self.nodes[j - 1]
    }

    /// The root `q_1` (full coefficient vector, size `N`).
    #[inline]
    pub fn root(&self) -> &[F] {
        &self.nodes[0]
    }

    /// Leaf layer: `q_N, …, q_{2N-1}`, each of size 1.
    pub fn leaves(&self) -> &[Vec<F>] {
        &self.nodes[self.big_n - 1..2 * self.big_n - 1]
    }

    /// Layer `i` (0-based): slice of node vectors at depth `i`.
    pub fn layer(&self, i: usize) -> &[Vec<F>] {
        assert!(i <= self.n, "layer {i} out of range [0, {}]", self.n);
        let start = (1usize << i) - 1;
        let end = (1usize << (i + 1)) - 1;
        &self.nodes[start..end]
    }

    /// Total field elements stored: `(n+1) · N`.
    pub fn total_field_elements(&self) -> usize {
        self.nodes.iter().map(|v| v.len()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `base^exp` via square-and-multiply.
fn field_pow<F: Field>(mut base: F, mut exp: usize) -> F {
    let mut result = F::one();
    while exp > 0 {
        if exp & 1 == 1 {
            result *= base;
        }
        base *= base;
        exp >>= 1;
    }
    result
}

/// Build the power table `[r^1, r^2, r^4, …, r^{2^{n-1}}]`.
///
/// Cost: `n - 1` field squarings (for `n ≥ 1`).
/// For `n = 0` returns an empty vector.
///
/// Used by `eval_estrin`, `eval_circuit`, and `eval_circuit_parallel`.
fn power_table<F: Field>(r: F, n: usize) -> Vec<F> {
    if n == 0 {
        return vec![];
    }
    let mut powers = Vec::with_capacity(n);
    powers.push(r); // r^{2^0} = r
    for k in 1..n {
        let prev = powers[k - 1];
        powers.push(prev * prev); // r^{2^k} = (r^{2^{k-1}})²
    }
    powers
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::Zero;

    fn fr(n: u64) -> Fr {
        Fr::from(n)
    }

    // ── UniTerm ───────────────────────────────────────────────────────────────

    #[test]
    fn term_degree_zero_is_constant() {
        let t = UniTerm::new(fr(7), 0);
        assert_eq!(t.eval(fr(5)), fr(7));
        assert_eq!(t.eval(fr(0)), fr(7));
    }

    #[test]
    fn term_eval_degree_one() {
        assert_eq!(UniTerm::new(fr(3), 1).eval(fr(4)), fr(12));
    }

    #[test]
    fn term_eval_degree_two() {
        assert_eq!(UniTerm::new(fr(5), 2).eval(fr(3)), fr(45));
    }

    #[test]
    fn term_eval_at_zero() {
        assert_eq!(UniTerm::new(fr(99), 3).eval(fr(0)), fr(0));
    }

    #[test]
    fn term_eval_at_one() {
        assert_eq!(UniTerm::new(fr(42), 7).eval(fr(1)), fr(42));
    }

    #[test]
    fn term_zero_detection() {
        assert!(UniTerm::<Fr>::new(Fr::zero(), 5).is_zero());
        assert!(!UniTerm::<Fr>::new(fr(1), 5).is_zero());
    }

    #[test]
    fn term_zero_coeff_evals_to_zero() {
        assert_eq!(UniTerm::<Fr>::new(Fr::zero(), 10).eval(fr(123)), Fr::zero());
    }

    #[test]
    fn term_eval_monomial() {
        assert_eq!(UniTerm::new(fr(1), 3).eval_monomial(fr(2)), fr(8));
    }

    // ── UniPoly construction and padding ──────────────────────────────────────

    #[test]
    fn new_power_of_two_no_padding() {
        let p = UniPoly::new(vec![fr(1); 8]);
        assert_eq!(p.padded_len(), 8);
        assert_eq!(p.log_len(), 3);
    }

    #[test]
    fn new_pads_to_next_power_of_two() {
        assert_eq!(UniPoly::new(vec![fr(1); 5]).padded_len(), 8);
        assert_eq!(UniPoly::new(vec![fr(1); 6]).padded_len(), 8);
        assert_eq!(UniPoly::new(vec![fr(1); 3]).padded_len(), 4);
    }

    #[test]
    fn new_length_1_stays_1() {
        assert_eq!(UniPoly::new(vec![fr(5)]).padded_len(), 1);
    }

    #[test]
    #[should_panic]
    fn new_panics_on_empty() {
        UniPoly::<Fr>::new(vec![]);
    }

    #[test]
    fn zero_poly_is_zero() {
        assert!(UniPoly::<Fr>::zero(3).is_zero_poly());
    }

    #[test]
    fn padding_zeros_are_transparent_to_coeff() {
        let p = UniPoly::new(vec![fr(1), fr(2), fr(3)]);
        assert_eq!(p.coeff(3), Fr::zero());
        assert_eq!(p.coeff(99), Fr::zero());
    }

    // ── degree ────────────────────────────────────────────────────────────────

    #[test]
    fn degree_constant() {
        let mut c = vec![Fr::zero(); 4];
        c[0] = fr(5);
        assert_eq!(UniPoly::new(c).degree(), 0);
    }

    #[test]
    fn degree_last_nonzero_before_padding() {
        assert_eq!(UniPoly::new(vec![fr(1), fr(0), fr(3)]).degree(), 2);
    }

    #[test]
    fn degree_zero_poly() {
        assert_eq!(UniPoly::<Fr>::zero(3).degree(), 0);
    }

    // ── terms ─────────────────────────────────────────────────────────────────

    #[test]
    fn terms_excludes_padding_zeros() {
        let p = UniPoly::new(vec![fr(2), fr(0), fr(5)]);
        let t: Vec<_> = p.terms().collect();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], UniTerm::new(fr(2), 0));
        assert_eq!(t[1], UniTerm::new(fr(5), 2));
    }

    #[test]
    fn all_terms_includes_padding() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3)]).all_terms().count(),
            4
        );
    }

    // ── eval_naive ────────────────────────────────────────────────────────────

    #[test]
    fn eval_naive_at_zero_returns_constant() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_naive(fr(0)),
            fr(1)
        );
    }

    #[test]
    fn eval_naive_at_one_returns_sum() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_naive(fr(1)),
            fr(10)
        );
    }

    #[test]
    fn eval_naive_linear() {
        assert_eq!(UniPoly::new(vec![fr(3), fr(5)]).eval_naive(fr(2)), fr(13));
    }

    // ── eval_horner ───────────────────────────────────────────────────────────

    #[test]
    fn eval_horner_at_zero() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_horner(fr(0)),
            fr(1)
        );
    }

    #[test]
    fn eval_horner_at_one() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_horner(fr(1)),
            fr(10)
        );
    }

    #[test]
    fn eval_horner_linear() {
        assert_eq!(UniPoly::new(vec![fr(3), fr(5)]).eval_horner(fr(2)), fr(13));
    }

    // ── power_table ───────────────────────────────────────────────────────────

    #[test]
    fn power_table_n3() {
        // r=2: powers[0]=r^1=2, powers[1]=r^2=4, powers[2]=r^4=16
        let pt = power_table(fr(2), 3);
        assert_eq!(pt.len(), 3);
        assert_eq!(pt[0], fr(2)); // r^1 = 2
        assert_eq!(pt[1], fr(4)); // r^2 = 4
        assert_eq!(pt[2], fr(16)); // r^4 = 16
    }

    #[test]
    fn power_table_n0_is_empty() {
        assert!(power_table(fr(5), 0).is_empty());
    }

    #[test]
    fn power_table_each_entry_is_square_of_previous() {
        let pt = power_table(fr(3), 5);
        for k in 1..5 {
            assert_eq!(pt[k], pt[k - 1] * pt[k - 1]);
        }
    }

    // ── eval_estrin ───────────────────────────────────────────────────────────

    #[test]
    fn eval_estrin_at_zero() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_estrin(fr(0)),
            fr(1)
        );
    }

    #[test]
    fn eval_estrin_at_one() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_estrin(fr(1)),
            fr(10)
        );
    }

    #[test]
    fn eval_estrin_linear() {
        assert_eq!(UniPoly::new(vec![fr(3), fr(5)]).eval_estrin(fr(2)), fr(13));
    }

    // ── eval_circuit ──────────────────────────────────────────────────────────

    #[test]
    fn eval_circuit_at_zero() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_circuit(fr(0)),
            fr(1)
        );
    }

    #[test]
    fn eval_circuit_at_one() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_circuit(fr(1)),
            fr(10)
        );
    }

    #[test]
    fn eval_circuit_linear() {
        assert_eq!(UniPoly::new(vec![fr(3), fr(5)]).eval_circuit(fr(2)), fr(13));
    }

    // ── eval_circuit_parallel ─────────────────────────────────────────────────

    #[test]
    fn eval_circuit_parallel_at_zero() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_circuit_parallel(fr(0)),
            fr(1)
        );
    }

    #[test]
    fn eval_circuit_parallel_at_one() {
        assert_eq!(
            UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4)]).eval_circuit_parallel(fr(1)),
            fr(10)
        );
    }

    // ── All four methods agree ────────────────────────────────────────────────

    #[test]
    fn all_methods_agree_n3() {
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let p = UniPoly::new(coeffs);
        for r in [fr(0), fr(1), fr(2), fr(7), fr(100)] {
            let naive = p.eval_naive(r);
            let horner = p.eval_horner(r);
            let estrin = p.eval_estrin(r);
            let circuit = p.eval_circuit(r);
            let parallel = p.eval_circuit_parallel(r);
            assert_eq!(naive, horner, "naive vs horner   at r={r}");
            assert_eq!(naive, estrin, "naive vs estrin   at r={r}");
            assert_eq!(naive, circuit, "naive vs circuit  at r={r}");
            assert_eq!(naive, parallel, "naive vs parallel at r={r}");
        }
    }

    #[test]
    fn all_methods_agree_n4() {
        let coeffs: Vec<Fr> = (0..16).map(|i| fr(i as u64 * 3 + 1)).collect();
        let p = UniPoly::new(coeffs);
        for r in [fr(1), fr(3), fr(11), fr(255)] {
            let naive = p.eval_naive(r);
            let circuit = p.eval_circuit(r);
            let estrin = p.eval_estrin(r);
            let par = p.eval_circuit_parallel(r);
            assert_eq!(naive, circuit, "circuit  at r={r}");
            assert_eq!(naive, estrin, "estrin   at r={r}");
            assert_eq!(naive, par, "parallel at r={r}");
        }
    }

    #[test]
    fn all_methods_agree_non_power_of_two_input() {
        // length 5 → padded to 8
        let coeffs: Vec<Fr> = (1..=5).map(fr).collect();
        let p = UniPoly::new(coeffs);
        for r in [fr(0), fr(1), fr(3), fr(11)] {
            let naive = p.eval_naive(r);
            let horner = p.eval_horner(r);
            let circuit = p.eval_circuit(r);
            assert_eq!(naive, horner, "at r={r}");
            assert_eq!(naive, circuit, "at r={r}");
        }
    }

    #[test]
    fn padding_does_not_change_eval_circuit() {
        // f = 1 + 2X + 3X² (length 3 → padded to 4)
        let p_short = UniPoly::new(vec![fr(1), fr(2), fr(3)]);
        let p_padded = UniPoly::new(vec![fr(1), fr(2), fr(3), fr(0)]);
        for r in [fr(0), fr(1), fr(2), fr(7)] {
            assert_eq!(p_short.eval_circuit(r), p_padded.eval_circuit(r));
        }
    }

    // ── decompose ─────────────────────────────────────────────────────────────

    #[test]
    fn decompose_node_count() {
        let d = UniPoly::new((0..8).map(fr).collect()).decompose();
        assert_eq!(d.nodes.len(), 15);
    }

    #[test]
    fn decompose_root_equals_input() {
        let coeffs: Vec<Fr> = (1..=8).map(fr).collect();
        let p = UniPoly::new(coeffs.clone());
        assert_eq!(p.decompose().root(), coeffs.as_slice());
    }

    #[test]
    fn decompose_layer_sizes() {
        let d = UniPoly::new((0..8).map(fr).collect()).decompose();
        assert_eq!(d.layer(0).len(), 1);
        assert_eq!(d.layer(1).len(), 2);
        assert_eq!(d.layer(2).len(), 4);
        assert_eq!(d.layer(3).len(), 8);
    }

    #[test]
    fn decompose_total_field_elements() {
        let d = UniPoly::new((0..8).map(fr).collect()).decompose();
        assert_eq!(d.total_field_elements(), (d.n + 1) * d.big_n);
    }

    #[test]
    fn decompose_first_layer_paper_example_n3() {
        let p = UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)]);
        let d = p.decompose();
        assert_eq!(d.q(2), &[fr(1), fr(3), fr(5), fr(7)]);
        assert_eq!(d.q(3), &[fr(2), fr(4), fr(6), fr(8)]);
    }

    #[test]
    fn decompose_second_layer_paper_example_n3() {
        let p = UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)]);
        let d = p.decompose();
        assert_eq!(d.q(4), &[fr(1), fr(5)]);
        assert_eq!(d.q(5), &[fr(3), fr(7)]);
        assert_eq!(d.q(6), &[fr(2), fr(6)]);
        assert_eq!(d.q(7), &[fr(4), fr(8)]);
    }

    #[test]
    fn decompose_leaves_paper_example_n3() {
        let p = UniPoly::new(vec![fr(1), fr(2), fr(3), fr(4), fr(5), fr(6), fr(7), fr(8)]);
        let d = p.decompose();
        assert_eq!(d.q(8), &[fr(1)]);
        assert_eq!(d.q(9), &[fr(5)]);
        assert_eq!(d.q(10), &[fr(3)]);
        assert_eq!(d.q(11), &[fr(7)]);
        assert_eq!(d.q(12), &[fr(2)]);
        assert_eq!(d.q(13), &[fr(6)]);
        assert_eq!(d.q(14), &[fr(4)]);
        assert_eq!(d.q(15), &[fr(8)]);
    }

    #[test]
    fn decompose_gate_relation_holds_n3() {
        let p = UniPoly::new((1..=8).map(fr).collect());
        let d = p.decompose();
        for j in 1..d.big_n {
            let qj = d.q(j);
            let left = d.q(2 * j);
            let right = d.q(2 * j + 1);
            for r in 0..qj.len() / 2 {
                assert_eq!(qj[2 * r], left[r], "j={j} left  r={r}");
                assert_eq!(qj[2 * r + 1], right[r], "j={j} right r={r}");
            }
        }
    }

    #[test]
    fn decompose_padded_input() {
        let p = UniPoly::new(vec![fr(1), fr(2), fr(3)]);
        let d = p.decompose();
        assert_eq!(d.q(2), &[fr(1), fr(3)]);
        assert_eq!(d.q(3), &[fr(2), fr(0)]);
    }
}
