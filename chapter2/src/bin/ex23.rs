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
    
    for k in 0..(N - 1) {
        let (i, _pivot) = (k..N)
            .map(|i| (i, u[(i, k)]))
            .filter(|(_, value)| value.abs() > EPSILON)
            .max_by(|(_, a), (_, b)| f64::partial_cmp(&a.abs(), &b.abs()).expect("found NaN or Inf"))
            .expect("Matrix is not singular");
        
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

