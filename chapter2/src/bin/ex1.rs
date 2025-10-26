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

    let stats = (0..100)
        .map(|_| dbg!(solver.experiment_randomly()))
        .collect::<Vec<_>>();

//     let residual_norm_plot_name = format!("plot/ex1/n{N}-residual_norm.svg");
//     let relative_error_plot_name = format!("plot/ex1/n{N}-relative_error.svg");
//     let time_elapsed_plot_name = format!("plot/ex1/n{N}-time_elapsed.svg");
//         
//         let root = SVGBackend::new(&residual_norm_plot_name, (800, 600)).into_drawing_area();
//         root.fill(&WHITE)?;
//         
//         let mut chart = plotters::chart::ChartBuilder::on(&root)
//             .caption(format!("消去法による残差ノルム (n = {N})"), ("sans-serif", 20).into_font())
//             .margin(10)
//             .x_label_area_size(40)
//             .y_label_area_size(40)
//             .build_cartesian_2d(
//                 (-5..105).with_key_points(vec![0, 20, 40, 60, 80, 100]),
//                 (1e-6..1e-3).log_scale().with_key_points(vec![1e-6, 1e-5, 1e-4, 1e-3, 1e-3]),
//             )?;
//         
//         chart.configure_mesh()
//             .x_desc("trials")
//             .y_desc("residual norm")
//             .y_label_formatter(&format_y_label)
//             .axis_desc_style(("sans-serif", 16).into_font())
//             .label_style(("sans-serif", 16).into_font())
//             .draw()?;
//         
//         root.present()?;    

//     MakePlot {
//         caption: format!("消去法による残差ノルム (n = {N})"),
//         y_desc: "residual norm",
//         y_coord: (1e-6..1e-3).log_scale().with_key_points(vec![1e-6, 1e-5, 1e-4, 1e-3]),
//     }

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
