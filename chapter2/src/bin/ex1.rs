use chapter2::{Matrix, Vector};
use chapter2::{EPSILON, EquationSolver, EquationExperimentStat, back_substitution};

fn do_gaussian_elimination_for_n_n1(ab: &mut nalgebra::DMatrix<f64>) {
    let n = ab.nrows();
    let m = ab.ncols();
    
    assert!(n + 1 == m);
    
    for k in 0..(n - 1) {
        let (i, _pivot) = (k..n)
            .map(|i| (i, ab[(i, k)]))
            .filter(|(_, value)| value.abs() > EPSILON)
            .max_by(|(_, a), (_, b)| f64::partial_cmp(&a.abs(), &b.abs()).expect("found NaN or Inf"))
            .expect("Matrix is singular");
        
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

fn solve_by_gaussian_elimination<const N: usize>(a: &Matrix<N, N>, b: &Vector<N>) -> Vector<N> {
    let mut augmented_coefficient_matrix = nalgebra::DMatrix::from_fn(N, N + 1,|i, j| {
        if j < N { a[(i, j)] } else { b[i] }
    });
    
    do_gaussian_elimination_for_n_n1(&mut augmented_coefficient_matrix);
    
    back_substitution(
        &Matrix::<N, N>::from_fn(|i, j| augmented_coefficient_matrix[(i, j)]),
        &Vector::<N>::from_fn(|i, _| augmented_coefficient_matrix[(i, N)]),
    )
}

fn plot_100_experiments<const N: usize>(solver: EquationSolver<N>) -> Result<(), Box<dyn std::error::Error>> {
    use chapter2::{Plotter, PlotterInit, IntoLogRange as _, BindKeyPoints as _};
    
    let stats: [EquationExperimentStat<N>; 100] = (0..100)
        .map(|_| dbg!(solver.experiment_randomly()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    
    let mut p = Plotter::init(PlotterInit {
        caption: format!("消去法による残差ノルム (n = {N})"),
        y_desc: "residual norm",
        y_coord: (1e-6..1e-3).log_scale().with_key_points(vec![1e-6, 1e-5, 1e-4, 1e-3, 1e-3]),
    });
    p.plot(stats.iter().map(|stat| stat.residual_norm).collect::<Vec<_>>().try_into().unwrap());
    p.write_into(format!("plot/ex1/n{N}-residual_norm.svg"))?;
    
    let mut p = Plotter::init(PlotterInit {
        caption: format!("消去法による相対誤差 (n = {N})"),
        y_desc: "relative error",
        y_coord: (1e-6..1e-3).log_scale().with_key_points(vec![1e-6, 1e-5, 1e-4, 1e-3]),
    });
    p.plot(stats.iter().map(|stat| stat.relative_error).collect::<Vec<_>>().try_into().unwrap());
    p.write_into(format!("plot/ex1/n{N}-relative_error.svg"))?;
    
    let mut p = Plotter::init(PlotterInit {
        caption: format!("消去法による計算時間 (n = {N})"),
        y_desc: "time elapsed (sec.)",
        y_coord: (1e-3..4.*1e-3),
    });
    p.plot(stats.iter().map(|stat| stat.elapsed.as_secs_f64()).collect::<Vec<_>>().try_into().unwrap());
    p.write_into(format!("plot/ex1/n{N}-time_elapsed.svg"))?;

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
        let mut ab = nalgebra::DMatrix::from_row_slice(3, 4, &[
            2.0, 1.0, -1.0, 8.0,
            -3.0, -1.0, 2.0, -11.0,
            -2.0, 1.0, 2.0, -3.0,
        ]);
        
        do_gaussian_elimination_for_n_n1(&mut ab);
        
        dbg!(ab.as_slice());
        
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
