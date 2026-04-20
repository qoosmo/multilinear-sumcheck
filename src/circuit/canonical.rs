use ark_ff::Field;

use crate::poly::{CanonicalPoly, MlPoly};
use super::bit_reverse_cache;

pub fn bit_reverse(k: usize, n: usize) -> usize {
    let mut k = k;
    let mut result = 0usize;
    for _ in 0..n {
        result = (result << 1) | (k & 1);
        k >>= 1;
    }
    result
}

pub fn build_bit_reverse_table(n: usize) -> Vec<usize> {
    let big_n = 1usize << n;
    (0..big_n).map(|k| bit_reverse(k, n)).collect()
}

pub struct CanonicalDecomp<F: Field> {
    pub n: usize,
    pub big_n: usize,
    pub nodes: Vec<CanonicalPoly<F>>,
    pub bit_rev_table: Vec<usize>,
}

impl<F: Field> CanonicalDecomp<F> {
    pub fn build(f: &CanonicalPoly<F>) -> Self {
        let n     = f.num_vars();
        let big_n = f.num_evals();

        let mut nodes: Vec<Option<CanonicalPoly<F>>> = vec![None; 2 * big_n - 1];
        nodes[0] = Some(f.clone());

        for i in 0..n {
            let layer_start = 1usize << i;
            let layer_end   = 1usize << (i + 1);

            for j in layer_start..layer_end {
                let qj    = nodes[j - 1].take().expect("node must be initialised");
                let beta  = qj.coeffs().to_vec();
                let half  = beta.len() / 2;

                let left_coeffs:  Vec<F> = (0..half).map(|r| beta[2 * r]).collect();
                let right_coeffs: Vec<F> = (0..half).map(|r| beta[2 * r + 1]).collect();

                nodes[j - 1]     = Some(qj);
                nodes[2 * j - 1] = Some(CanonicalPoly::new(left_coeffs));
                nodes[2 * j]     = Some(CanonicalPoly::new(right_coeffs));
            }
        }

        let nodes: Vec<CanonicalPoly<F>> = nodes
            .into_iter()
            .enumerate()
            .map(|(idx, opt)| {
                opt.unwrap_or_else(|| panic!("node q_{} was not filled", idx + 1))
            })
            .collect();

        let bit_rev_table = bit_reverse_cache::get_or_build(n).into_owned();

        Self { n, big_n, nodes, bit_rev_table }
    }

    #[inline]
    pub fn q(&self, j: usize) -> &CanonicalPoly<F> {
        assert!(j >= 1 && j <= 2 * self.big_n - 1,
            "q index {j} out of range [1, {}]", 2 * self.big_n - 1);
        &self.nodes[j - 1]
    }

    pub fn root(&self) -> &CanonicalPoly<F> {
        &self.nodes[0]
    }

    pub fn leaves(&self) -> &[CanonicalPoly<F>] {
        &self.nodes[self.big_n - 1 .. 2 * self.big_n - 1]
    }

    pub fn layer(&self, i: usize) -> &[CanonicalPoly<F>] {
        assert!(i <= self.n, "layer {i} out of range [0, {}]", self.n);
        let start = (1usize << i) - 1;
        let end   = (1usize << (i + 1)) - 1;
        &self.nodes[start..end]
    }

    pub fn total_field_elements(&self) -> usize {
        self.nodes.iter().map(|q| q.num_evals()).sum()
    }

    pub fn leaves_are_bit_reverse_of_root(&self) -> bool {
        let root_coeffs = self.root().coeffs();
        let leaves      = self.leaves();
        (0..self.big_n).all(|k| {
            let rev_k = self.bit_rev_table[k];
            leaves[k].coeffs()[0] == root_coeffs[rev_k]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::MlPoly;
    use ark_bn254::Fr;

    fn fr(n: u64) -> Fr { Fr::from(n) }

    #[test]
    fn bit_reverse_n3_cases() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
        assert_eq!(bit_reverse(0b101, 3), 0b101);
        assert_eq!(bit_reverse(0b110, 3), 0b011);
        assert_eq!(bit_reverse(0b111, 3), 0b111);
    }

    #[test]
    fn bit_reverse_is_involution() {
        for n in 1..=8 {
            for k in 0..(1usize << n) {
                assert_eq!(bit_reverse(bit_reverse(k, n), n), k);
            }
        }
    }

    #[test]
    fn bit_reverse_table_length() {
        assert_eq!(build_bit_reverse_table(4).len(), 16);
    }

    #[test]
    fn node_count_is_2n_minus_1() {
        let f = CanonicalPoly::new((0..8).map(|i| fr(i)).collect());
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.nodes.len(), 15);
    }

    #[test]
    fn root_equals_input() {
        let coeffs: Vec<Fr> = (0..8).map(|i| fr(i)).collect();
        let f = CanonicalPoly::new(coeffs.clone());
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.root().coeffs(), f.coeffs());
    }

    #[test]
    fn total_field_elements_is_n_plus_1_times_n() {
        let f = CanonicalPoly::new((0..8).map(|i| fr(i)).collect());
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.total_field_elements(), (d.n + 1) * d.big_n);
    }

    #[test]
    fn layer_sizes_are_correct() {
        let f = CanonicalPoly::new((0..8).map(|i| fr(i)).collect());
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.layer(0).len(), 1);
        assert_eq!(d.layer(1).len(), 2);
        assert_eq!(d.layer(2).len(), 4);
        assert_eq!(d.layer(3).len(), 8);
    }

    #[test]
    fn node_coeff_sizes_decrease_by_layer() {
        let f = CanonicalPoly::new((0..8).map(|i| fr(i)).collect());
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.q(1).num_evals(), 8);
        assert_eq!(d.q(2).num_evals(), 4);
        assert_eq!(d.q(3).num_evals(), 4);
        assert_eq!(d.q(4).num_evals(), 2);
        assert_eq!(d.q(8).num_evals(), 1);
    }

    #[test]
    fn split_matches_paper_example_n3() {
        let coeffs = vec![fr(1),fr(2),fr(3),fr(4),fr(5),fr(6),fr(7),fr(8)];
        let f = CanonicalPoly::new(coeffs);
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.q(2).coeffs(), &[fr(1),fr(3),fr(5),fr(7)]);
        assert_eq!(d.q(3).coeffs(), &[fr(2),fr(4),fr(6),fr(8)]);
    }

    #[test]
    fn second_layer_split_correct_n3() {
        let coeffs = vec![fr(1),fr(2),fr(3),fr(4),fr(5),fr(6),fr(7),fr(8)];
        let f = CanonicalPoly::new(coeffs);
        let d = CanonicalDecomp::build(&f);
        assert_eq!(d.q(4).coeffs(), &[fr(1),fr(5)]);
        assert_eq!(d.q(5).coeffs(), &[fr(3),fr(7)]);
        assert_eq!(d.q(6).coeffs(), &[fr(2),fr(6)]);
        assert_eq!(d.q(7).coeffs(), &[fr(4),fr(8)]);
    }

    #[test]
    fn leaves_are_bit_reverse_permutation_of_root_n3() {
        let f = CanonicalPoly::new((1..=8).map(|i| fr(i)).collect());
        let d = CanonicalDecomp::build(&f);
        assert!(d.leaves_are_bit_reverse_of_root());
    }

    #[test]
    fn leaves_are_bit_reverse_permutation_n1() {
        let f = CanonicalPoly::new(vec![fr(3), fr(7)]);
        let d = CanonicalDecomp::build(&f);
        assert!(d.leaves_are_bit_reverse_of_root());
    }

    #[test]
    fn leaves_are_bit_reverse_permutation_n4() {
        let f = CanonicalPoly::new((0..16).map(|i| fr(i as u64 * 3 + 1)).collect());
        let d = CanonicalDecomp::build(&f);
        assert!(d.leaves_are_bit_reverse_of_root());
    }

    #[test]
    fn gate_relation_holds_for_all_internal_nodes_n3() {
        let f = CanonicalPoly::new((0..8).map(|i| fr(i as u64 + 1)).collect());
        let d = CanonicalDecomp::build(&f);
        for j in 1..d.big_n {
            let qj      = d.q(j).coeffs().to_vec();
            let q_left  = d.q(2 * j).coeffs().to_vec();
            let q_right = d.q(2 * j + 1).coeffs().to_vec();
            let m = qj.len();
            for r in 0..m / 2 {
                assert_eq!(qj[2 * r],     q_left[r],  "j={j} even r={r}");
                assert_eq!(qj[2 * r + 1], q_right[r], "j={j} odd  r={r}");
            }
        }
    }
}