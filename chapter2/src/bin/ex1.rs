use chapter2::{Solver, back_substitution};
use nalgebra::{DMatrix, SMatrix, SVector};

fn do_gaussian_elimination(ab: &mut DMatrix<f64>) {
    let n = ab.nrows();
    let m = ab.ncols();
    
    assert!(n + 1 == m);
    
    for k in 0..(n - 1) {
        let (i, pivot) = ab
            .column(k)
            .iter()
            .enumerate()
            .skip(k)
            .map(|(i, x)| (i, x.abs()))
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or_else(|| panic!("cannot compare {x} and {y}")))
            .unwrap();
        
        assert!(pivot != 0.0, "Matrix is singular.");
        
        if i != k {
            ab.swap_rows(i, k);
        }
        
        for i in (k + 1)..n {
            let factor = ab[(i, k)] / ab[(k, k)];
            for j in k..m {
                ab[(i, j)] -= factor * ab[(k, j)];
            }
        }
    }
}

#[cfg(test)]
#[test]
fn test_do_gaussian_elimination() {
    let mut ab = DMatrix::from_row_slice(3, 4, &[
        2.0, 1.0, -1.0, 8.0,
        -3.0, -1.0, 2.0, -11.0,
        -2.0, 1.0, 2.0, -3.0,
    ]);
    
    do_gaussian_elimination(&mut ab);
    
    dbg!(ab.as_slice());
    
    let expected = DMatrix::from_row_slice(3, 4, &[
        -3.0, -1.0, 2.0, -11.0,
        0.0, 5./3., 2./3., 13./3.,
        0.0, 0.0, 1./5., -1./5.,
    ]);
    
    for i in 0..3 {
        for j in 0..4 {
            assert!(
                (ab[(i, j)] - expected[(i, j)]).abs() < 1e-10,
                "ab[{i}, {j}] = {}, expected[{i}, {j}] = {}", ab[(i, j)], expected[(i, j)]
            );
        }
    }
}

fn solve_by_gaussian_elimination<const N: usize>(a: SMatrix<f64, N, N>, b: SVector<f64, N>) -> SVector<f64, N> {
    let mut augmented_coefficient_matrix = DMatrix::from_fn(N, N + 1, |i, j| {
        if j < N { a[(i, j)] } else { b[i] }
    });
    
    do_gaussian_elimination(&mut augmented_coefficient_matrix);
    
    back_substitution(
        &SMatrix::from_fn(|i, j| augmented_coefficient_matrix[(i, j)]),
        &SVector::from_fn(|i, _| augmented_coefficient_matrix[(i, N)]),
    )
}

fn experiment<const N: usize>() -> chapter2::ExperimentResult<N> {
    Solver::new(solve_by_gaussian_elimination).experiment_randomly()
}

fn main() {
    for _ in 0..100 {
        // let ExperimentResult {
        //     my_solution,
        //     reference_solution,
        //     residual_norm,
        //     relative_error,
        //     elapsed,
        // } =
        dbg!(experiment::<100>());
    }
}
