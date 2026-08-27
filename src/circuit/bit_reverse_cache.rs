use std::borrow::Cow;
use std::sync::OnceLock;

use super::build_bit_reverse_table;

const fn const_bit_reverse(mut k: usize, n: usize) -> usize {
    let mut result = 0usize;
    let mut i = 0;
    while i < n {
        result = (result << 1) | (k & 1);
        k >>= 1;
        i += 1;
    }
    result
}

macro_rules! make_table {
    ($n:expr) => {{
        const N: usize = 1 << $n;
        let mut table = [0usize; N];
        let mut k = 0usize;
        while k < N {
            table[k] = const_bit_reverse(k, $n);
            k += 1;
        }
        table
    }};
}

pub static BIT_REV_N10: [usize; 1 << 10] = make_table!(10);
pub static BIT_REV_N15: [usize; 1 << 15] = make_table!(15);

static BIT_REV_N20_CELL: OnceLock<Vec<usize>> = OnceLock::new();

pub fn bit_rev_n20() -> &'static [usize] {
    BIT_REV_N20_CELL.get_or_init(|| build_bit_reverse_table(20))
}

pub fn get_bit_rev_table(n: usize) -> Option<&'static [usize]> {
    match n {
        10 => Some(&BIT_REV_N10),
        15 => Some(&BIT_REV_N15),
        20 => Some(bit_rev_n20()),
        _ => None,
    }
}

pub fn get_or_build(n: usize) -> Cow<'static, [usize]> {
    get_bit_rev_table(n)
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(build_bit_reverse_table(n)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_lengths_are_correct() {
        assert_eq!(BIT_REV_N10.len(), 1 << 10);
        assert_eq!(BIT_REV_N15.len(), 1 << 15);
        assert_eq!(bit_rev_n20().len(), 1 << 20);
    }

    #[test]
    fn n10_matches_runtime() {
        assert_eq!(&BIT_REV_N10[..], build_bit_reverse_table(10).as_slice());
    }

    #[test]
    fn n15_matches_runtime() {
        assert_eq!(&BIT_REV_N15[..], build_bit_reverse_table(15).as_slice());
    }

    #[test]
    fn n20_matches_runtime() {
        assert_eq!(bit_rev_n20(), build_bit_reverse_table(20).as_slice());
    }

    #[test]
    fn n10_is_involution() {
        for k in 0..(1 << 10) {
            assert_eq!(BIT_REV_N10[BIT_REV_N10[k]], k);
        }
    }

    #[test]
    fn n15_is_involution() {
        for k in 0..(1 << 15) {
            assert_eq!(BIT_REV_N15[BIT_REV_N15[k]], k);
        }
    }

    #[test]
    fn n20_is_involution() {
        let t = bit_rev_n20();
        for k in 0..(1 << 20) {
            assert_eq!(t[t[k]], k);
        }
    }

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(BIT_REV_N10[0], 0);
        assert_eq!(BIT_REV_N15[0], 0);
        assert_eq!(bit_rev_n20()[0], 0);
    }

    #[test]
    fn all_ones_maps_to_itself() {
        assert_eq!(BIT_REV_N10[(1 << 10) - 1], (1 << 10) - 1);
        assert_eq!(BIT_REV_N15[(1 << 15) - 1], (1 << 15) - 1);
        assert_eq!(bit_rev_n20()[(1 << 20) - 1], (1 << 20) - 1);
    }

    #[test]
    fn get_returns_some_for_cached_sizes() {
        assert!(get_bit_rev_table(10).is_some());
        assert!(get_bit_rev_table(15).is_some());
        assert!(get_bit_rev_table(20).is_some());
    }

    #[test]
    fn get_returns_none_for_uncached_sizes() {
        assert!(get_bit_rev_table(0).is_none());
        assert!(get_bit_rev_table(8).is_none());
        assert!(get_bit_rev_table(16).is_none());
        assert!(get_bit_rev_table(21).is_none());
    }

    #[test]
    fn get_or_build_cached_is_borrowed() {
        assert!(matches!(get_or_build(10), Cow::Borrowed(_)));
        assert!(matches!(get_or_build(15), Cow::Borrowed(_)));
        assert!(matches!(get_or_build(20), Cow::Borrowed(_)));
    }

    #[test]
    fn get_or_build_uncached_is_owned() {
        assert!(matches!(get_or_build(8), Cow::Owned(_)));
    }

    #[test]
    fn get_or_build_uncached_matches_runtime() {
        let cow = get_or_build(8);
        assert_eq!(cow.as_ref(), build_bit_reverse_table(8).as_slice());
    }
}
