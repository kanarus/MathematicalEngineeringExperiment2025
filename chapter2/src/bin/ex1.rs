#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use chapter2::{Matrix, Vector};
use chapter2::{EPSILON, EquationSolver, back_substitution};

fn do_gaussian_elimination<const N: usize>(ab: &mut Matrix<N, {N + 1}>) {
    for k in 0..(N - 1) {
        let (i, _pivot) = (k..N)
            .map(|i| (i, ab[(i, k)]))
            .filter(|(_, value)| value.abs() > EPSILON)
            .max_by(|(_, a), (_, b)| f64::partial_cmp(&a.abs(), &b.abs()).expect("found NaN or Inf"))
            .expect("Matrix is singular");
        
        if i != k {
            ab.swap_rows(i, k);
        }
        
        for i in (k + 1)..N {
            let factor = ab[(i, k)] / ab[(k, k)];
            for j in k..(N + 1) {
                ab[(i, j)] -= factor * ab[(k, j)];
            }
        }
    }
}

fn solve_by_gaussian_elimination<const N: usize>(a: &Matrix<N, N>, b: &Vector<N>) -> Vector<N> where [(); N + 1]: {
    let mut augmented_coefficient_matrix = Matrix::concat(a, b);
    do_gaussian_elimination(&mut augmented_coefficient_matrix);
    /*
     * rustc-1.92 reports error for
     * ```
     * let (a, b): (Matrix<N, N>, Vector<N>) = augmented_coefficient_matrix.into_split_last_column();
     * ```
     * as:
     * ```
     * mismatched types
     * expected constant N
     * found constant chapter2::::matrix::{impl#8}::into_split_last_column::{constant#0} (rustc E0308)
     * ```
    */
    back_substitution(
        &Matrix::<N, N>::from_fn(|i, j| augmented_coefficient_matrix[(i, j)]),
        &Vector::<N>::from_fn(|i, _| augmented_coefficient_matrix[(i, N)]),
    )
}

fn plot_100_experiments<const N: usize>(solver: EquationSolver<N>) -> Result<(), Box<dyn std::error::Error>> {
    let stats: [chapter2::EquationExperimentStat<N>; 100] = (0..100)
        .map(|_| dbg!(solver.experiment_randomly()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    
    chapter2::Plotter {
        y_desc: "residual norm",
        data: stats.iter().map(|stat| stat.residual_norm).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex1/n{N}-residual_norm.svg"))?;
    
    chapter2::Plotter {
        y_desc: "relative error",
        data: stats.iter().map(|stat| stat.relative_error).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex1/n{N}-relative_error.svg"))?;
    
    chapter2::Plotter {
        y_desc: "time elapsed (sec.)",
        data: stats.iter().map(|stat| stat.elapsed.as_secs_f64()).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex1/n{N}-time_elapsed.svg"))?;
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    plot_100_experiments(EquationSolver::new(solve_by_gaussian_elimination::<100>))?;
    plot_100_experiments(EquationSolver::new(solve_by_gaussian_elimination::<200>))?;
    plot_100_experiments(EquationSolver::new(solve_by_gaussian_elimination::<400>))?;
    plot_100_experiments(EquationSolver::new(solve_by_gaussian_elimination::<800>))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_do_gaussian_elimination() {
        let mut ab = Matrix::from([
            [2.0, 1.0, -1.0, 8.0],
            [-3.0, -1.0, 2.0, -11.0],
            [-2.0, 1.0, 2.0, -3.0],
        ]);
        
        do_gaussian_elimination(&mut ab);
        
        dbg!(&ab);
        
        let expected = nalgebra::DMatrix::from_row_slice(3, 4, &[
            -3.0, -1.0, 2.0, -11.0,
            0.0, 5./3., 2./3., 13./3.,
            0.0, 0.0, 1./5., -1./5.,
        ]);
        
        for i in 0..3 {
            for j in 0..4 {
                assert!(
                    (ab[(i, j)] - expected[(i, j)]).abs() < EPSILON,
                    "ab[{i}, {j}] = {}, expected[{i}, {j}] = {}", ab[(i, j)], expected[(i, j)]
                );
            }
        }
    }
}
