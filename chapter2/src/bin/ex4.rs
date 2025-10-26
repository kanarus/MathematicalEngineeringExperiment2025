use chapter2::{Matrix, Vector};
use chapter2::{EPSILON, DominantEigenvalueSolver, DominantEigenvalueSolution};

fn solve_by_power_iteration<const N: usize>(a: &Matrix<N, N>) -> DominantEigenvalueSolution<N> {
    const MAX_ITERATIONS: usize = 1000_000;
    
    let mut mu = Vec::<f64>::new();
    let mut x_k = Vector::<N>::filled_with(1.0);
    for count in 1..MAX_ITERATIONS {
        let y_k = a * &x_k;
        
        let (i, _max_abs) = x_k
            .iter()
            .enumerate()
            .max_by(|(_, p), (_, q)| f64::partial_cmp(&p.abs(), &q.abs()).expect("found NaN or Inf"))
            .expect("Vector is zero");
        
        let mu_k = y_k[i] / x_k[i];
        if mu.last().is_some_and(|it| (it.abs() - mu_k.abs()).abs() < EPSILON) {
            return DominantEigenvalueSolution {
                eigenvalue: mu_k,
                eigenvector: x_k,
                iteration_count: count,
            };
        }
        
        x_k = y_k.normalized();
        mu.push(mu_k);
    }
    
    panic!("`mu` seems to diverge");
}

fn plot_100_experiments<const N: usize>(solver: DominantEigenvalueSolver<N>) -> Result<(), Box<dyn std::error::Error>> {
    let stats: [chapter2::DominantEigenvalueExperimentStat<N>; 100] = (0..100)
        .map(|_| dbg!(solver.experiment_randomly()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    
    chapter2::Plotter {
        y_desc: "residual norm",
        data: stats.iter().map(|stat| stat.residual_norm).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex4/n{N}-residual_norm.svg"))?;
    
    chapter2::Plotter {
        y_desc: "eigenvalue's relative error",
        data: stats.iter().map(|stat| stat.eigenvalue_relative_error).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex4/n{N}-eigenvalue_relative_error.svg"))?;
    
    chapter2::Plotter {
        y_desc: "eigenvector's relative error",
        data: stats.iter().map(|stat| stat.eigenvector_relative_error).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex4/n{N}-eigenvector_relative_error.svg"))?;
    
    chapter2::Plotter {
        y_desc: "time elapsed (sec.)",
        data: stats.iter().map(|stat| stat.elapsed.as_secs_f64()).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex4/n{N}-time_elapsed.svg"))?;
    
    chapter2::Plotter {
        y_desc: "# of steps",
        data: stats.iter().map(|stat| stat.iteration_count as f64).collect::<Vec<_>>().try_into().unwrap(),
    }.plot_into(format!("plot/ex4/n{N}-iteration_count.svg"))?;
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    plot_100_experiments(DominantEigenvalueSolver::new(solve_by_power_iteration::<50>))?;
    plot_100_experiments(DominantEigenvalueSolver::new(solve_by_power_iteration::<100>))?;
    plot_100_experiments(DominantEigenvalueSolver::new(solve_by_power_iteration::<200>))?;
    plot_100_experiments(DominantEigenvalueSolver::new(solve_by_power_iteration::<400>))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solve_by_power_iteration() {
        let a = Matrix::<3, 3>::from([
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ]);
        
        let solution = dbg!(solve_by_power_iteration(&a));
        
        assert!((solution.eigenvalue - (f64::sqrt(2.) + 2.)).abs() < EPSILON);
        assert!((solution.eigenvector.normalized() - Vector::<3>::from([
            1., f64::sqrt(2.), 1.
        ]).normalized()).norm() < EPSILON);
    }
}
