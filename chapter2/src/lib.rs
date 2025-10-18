use nalgebra::{DMatrix, DVector, SMatrix, SVector};

pub const EPSILON: f64 = 1e-10;

pub const RANDOM_RANGE: std::ops::RangeInclusive<f64> = -1.0..=1.0;

pub fn reference_solution<const N: usize>(
    a: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> DVector<f64> {
    DMatrix::from_fn(N, N, |i, j| a[(i, j)])
        .lu()
        .solve(&DMatrix::from_column_slice(N, 1, b.as_slice()))
        .unwrap()
        .column(0)
        .into()
}

/// Solve Ay = b by forward substitution:
/// 
/// ```text
/// y_i = (b_i - sum_{j=0}^{i-1} a_{ij} y_j) / a_{ii}
/// for i = 0, 1, ..., N-1
/// ```
pub fn forward_substitution<const N: usize>(
    lower_triangular_matrix: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> SVector<f64, N> {
    assert!(
        (0..N).all(|i| lower_triangular_matrix.column(i).iter().take(i).enumerate().all(|(j, x)| {dbg!(i, j); dbg!(x.abs()) < EPSILON})),
        "Matrix is not lower triangular"
    );
    
    let mut y = SVector::<f64, N>::zeros();
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lower_triangular_matrix[(i, j)] * y[j];
        }
        y[i] = (b[i] - sum) / lower_triangular_matrix[(i, i)];
    }
    y
}

/// Solve Ax = b by back substitution:
/// 
/// ```text
/// x_i = (b_i - sum_{j=i+1}^{n} a_{ij} x_j) / a_{ii}
/// for i = N-1, N-2, ..., 0
/// ```
pub fn back_substitution<const N: usize>(
    upper_triangular_matrix: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
) -> SVector<f64, N> {
    assert!(
        (0..N).all(|i| upper_triangular_matrix.column(i).iter().skip(i + 1).all(|x| x.abs() < EPSILON)),
        "Matrix is not upper triangular"
    );
    
    let mut x = SVector::<f64, N>::zeros();
    for i in (0..N).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..N {
            sum += upper_triangular_matrix[(i, j)] * x[j];
        }
        x[i] = (b[i] - sum) / upper_triangular_matrix[(i, i)];
    }
    x
}

pub struct Solver<const N: usize>(
    fn(SMatrix<f64, N, N>, SVector<f64, N>) -> SVector<f64, N>,
);

#[derive(Debug)]
pub struct ExperimentResult<const N: usize> {
    pub my_solution: SVector<f64, N>,
    pub reference_solution: DVector<f64>,
    pub residual_norm: f64,
    pub relative_error: f64,
    pub elapsed: std::time::Duration,
}

impl<const N: usize> Solver<N> {
    pub fn new(
        f: fn(SMatrix<f64, N, N>, SVector<f64, N>) -> SVector<f64, N>,
    ) -> Self {
        Self(f)
    }
    
    pub fn experiment_randomly(&self) -> ExperimentResult<N> {
        use rand::{Rng, rng};
        
        let a = SMatrix::<f64, N, N>::from_fn(|_, _| rng().random_range(RANDOM_RANGE));
        let b = SVector::<f64, N>::from_fn(|_, _| rng().random_range(RANDOM_RANGE));    
        
        let (my_solution, elapsed) = {
            let t = std::time::Instant::now();
            ((self.0)(a, b), t.elapsed())
        };
        
        let reference_solution = reference_solution(&a, &b);
        
        let residual_norm = (b - a * my_solution).norm();
        let relative_error = (my_solution - &reference_solution).norm() / reference_solution.norm();
        
        ExperimentResult {
            my_solution,
            reference_solution,
            residual_norm,
            relative_error,
            elapsed,
        }
    }
}
