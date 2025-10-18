use chapter2::{EPSILON, Solver, forward_substitution, back_substitution};
use nalgebra::{SMatrix, SVector};

struct LUDecomposition<const N: usize> {
    l: SMatrix<f64, N, N>,
    u: SMatrix<f64, N, N>,
    pi: [usize; N],
}

fn lu_decomposition<const N: usize>(
    a: SMatrix<f64, N, N>,
) -> LUDecomposition<N> {
    // initialize `pi` as an identity permutation
    let mut pi: [usize; N] = std::array::from_fn(|i| i);
    // initialize `l` as an identity matrix
    let mut l = SMatrix::<f64, N, N>::identity();
    // initialize `u` as `a` itself
    let mut u = a;
    
    /*
     * NOTE:
     * 
     * Our textbook illustrates this step as
     * iterating k from 1 to **N - 1** by 1-based index,
     * which is equivalent to iterating k from 0 to **N - 2** by 0-based index.
     * 
     * It's wrong. It should be iterating k from 0 to **N - 1** by 0-based index,
     * i.e., 1 to **N** by 1-based index.
     */
    for k in 0..N {
        let (i, _pivot) = (k..N)
            .map(|i| (i, u[(i, k)]))
            .filter(|(_, value)| value.abs() > EPSILON)
            .max_by(|(_, a), (_, b)| f64::partial_cmp(&a.abs(), &b.abs()).expect("found NaN or Inf"))
            .expect("Matrix is singular");
        
        if i != k {
            u.swap_rows(i, k);
            l.swap_rows(i, k);
            pi.swap(i, k);
        }
        
        for i in (k + 1)..N {
            let factor = u[(i, k)] / u[(k, k)];
            for j in k..N {
                u[(i, j)] -= factor * u[(k, j)];
            }
            l[(i, k)] = factor;
        }
        l[(k, k)] = 1.0;
        l.column_mut(k).rows_range_mut(0..k).apply(|it| *it = 0.0);
    }
    
    LUDecomposition { l, u, pi }
}

fn solve_by_lu_decomposition<const N: usize>(
    a: SMatrix<f64, N, N>,
    b: SVector<f64, N>,
) -> SVector<f64, N> {
    let LUDecomposition { l, u, pi } = lu_decomposition(a);
    
    // solve Ly = Pb by forward substitution
    let y = forward_substitution(&l, &SVector::from_fn(|i, _| b[pi[i]]));
    // solve Ux = y by back substitution
    back_substitution(&u, &y)
}

fn experiment<const N: usize>() -> chapter2::ExperimentResult<N> {
    Solver::new(solve_by_lu_decomposition).experiment_randomly()
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

#[cfg(test)]
#[test]
fn test_lu_decomposition() {
    let a = SMatrix::<f64, 3, 3>::from_row_slice(&[
        2.0, 1.0, -1.0,
        -3.0, -1.0, 2.0,
        -2.0, 1.0, 2.0,
    ]);
    
    let my_decomposition = lu_decomposition(a);
    
    let reference_decomposition = nalgebra::Matrix::lu(a);

    assert!(reference_decomposition.l().shape() == (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (my_decomposition.l[(i, j)] - reference_decomposition.l()[(i, j)]).abs() < EPSILON,
                "L matrix differs at ({i}, {j}): {} vs {}",
                my_decomposition.l[(i, j)],
                reference_decomposition.l()[(i, j)]
            );
        }
    }
    
    assert!(reference_decomposition.u().shape() == (3, 3));
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (my_decomposition.u[(i, j)] - reference_decomposition.u()[(i, j)]).abs() < EPSILON,
                "U matrix differs at ({i}, {j}): {} vs {}",
                my_decomposition.u[(i, j)],
                reference_decomposition.u()[(i, j)]
            );
        }
    }
}
