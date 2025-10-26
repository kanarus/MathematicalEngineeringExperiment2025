/// N-rows and M-columns matrix
#[derive(Clone)]
pub struct Matrix<const N: usize, const M: usize> {
    /// using `Vec` instead of array to avoid stack overflow for large matrices
    columns: Vec<Vec<f64>>,
}

pub type Vector<const N: usize> = Matrix<N, 1>;

impl<const N: usize, const M: usize> From<[[f64; M]; N]> for Matrix<N, M> {
    fn from(array_of_rows: [[f64; M]; N]) -> Self {
        Self::from_fn(|i, j| array_of_rows[i][j])
    }
}
impl<const N: usize, const M: usize> From<&[[f64; M]; N]> for Matrix<N, M> {
    fn from(array_of_rows: &[[f64; M]; N]) -> Self {
        Self::from_fn(|i, j| array_of_rows[i][j])
    }
}
impl<const N: usize, const M: usize> From<[&[f64; M]; N]> for Matrix<N, M> {
    fn from(array_of_rows: [&[f64; M]; N]) -> Self {
        Self::from_fn(|i, j| array_of_rows[i][j])
    }
}
impl<const N: usize, const M: usize> TryFrom<&[&[f64]]> for Matrix<N, M> {
    type Error = &'static str;
    fn try_from(slice_of_rows: &[&[f64]]) -> Result<Self, Self::Error> {
        (slice_of_rows.len() == N && slice_of_rows.iter().all(|row| row.len() == M))
            .then(|| Self::from_fn(|i, j| slice_of_rows[i][j]))
            .ok_or("slice of rows dimensions do not match matrix size")
    }
}

impl<const N: usize> From<[f64; N]> for Vector<N> {
    fn from(array: [f64; N]) -> Self {
        Self { columns: vec![array.to_vec()] }
    }
}
impl<const N: usize> From<&[f64; N]> for Vector<N> {
    fn from(array: &[f64; N]) -> Self {
        Self { columns: vec![array.to_vec()] }
    }
}
impl<const N: usize> TryFrom<&[f64]> for Vector<N> {
    type Error = &'static str;
    fn try_from(slice: &[f64]) -> Result<Self, Self::Error> {
        (slice.len() == N)
            .then(|| Self { columns: vec![slice.to_vec()] })
            .ok_or("slice length does not match vector size")
    }
}
impl<const N: usize> AsRef<[f64]> for Vector<N> {
    fn as_ref(&self) -> &[f64] {
        &self.columns[0]
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> f64) -> Self {
        Self { columns: (0..M).map(|j| (0..N).map(|i| f(i, j)).collect()).collect() }
    }
    
    pub fn filled_with(value: f64) -> Self {
        Self { columns: vec![vec![value; N]; M] }
    }
    
    pub fn zeroed() -> Self {
        Self::filled_with(0.0)
    }
    
    pub fn transpose(&self) -> Matrix<M, N> {
        Matrix::<M, N>::from_fn(|i, j| self[(j, i)])
    }
    
    pub fn concat<const L: usize>(a: &Matrix<N, M>, b: &Matrix<N, L>) -> Matrix<N, {M + L}> {
        Matrix::<N, {M + L}>::from_fn(|i, j| {
            if j < M {
                a[(i, j)]
            } else {
                b[(i, j - M)]
            }
        })
    }
    
    pub fn swap_rows(&mut self, i: usize, k: usize) {
        (0..M).for_each(|j| {
            self.columns[j].swap(i, k);
        });
    }
}

impl<const N: usize> Matrix<N, N> {
    pub fn identity() -> Self {
        Self::from_fn(|i, j| if i == j { 1.0 } else { 0.0 })
    }
}

impl<const N: usize> Vector<N> {
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.as_ref().iter()
    }
    
    pub fn norm(&self) -> f64 {
        self.as_ref().iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    
    pub fn normalize(&mut self) {
        let norm = self.norm();
        *self /= norm;
    }
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }
    
    pub fn dot(&self, rhs: &Self) -> f64 {
        let result_matrix: Matrix<1, 1> = self.transpose() * rhs;
        result_matrix[(0, 0)]
    }
}

const _: () = {
    pub struct Column<'a>(std::slice::Iter<'a, f64>);
    impl<'a> Iterator for Column<'a> {
        type Item = f64;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next().copied()
        }
    }
    
    pub struct ColumnMut<'a>(std::slice::IterMut<'a, f64>);
    impl<'a> Iterator for ColumnMut<'a> {
        type Item = &'a mut f64;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }
    }
    
    impl<const N: usize, const M: usize> Matrix<N, M> {
        pub fn column(&self, j: usize) -> Column<'_> {
            Column(self.columns[j].iter())
        }
        
        pub fn column_mut(&mut self, j: usize) -> ColumnMut<'_> {
            ColumnMut(self.columns[j].iter_mut())
        }
    }
};

impl<const N: usize, const M: usize> std::fmt::Debug for Matrix<N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if M == 1 {
            self.columns[0].fmt(f)
        } else {
            self.columns.fmt(f)
        }
    }
}

impl<const N: usize, const M: usize> std::ops::Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.columns[j][i]
    }
}
impl<const N: usize, const M: usize> std::ops::IndexMut<(usize, usize)> for Matrix<N, M> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.columns[j][i]
    }
}

impl<const N: usize> std::ops::Index<usize> for Vector<N> {
    type Output = f64;
    fn index(&self, i: usize) -> &Self::Output {
        &self.columns[0][i]
    }
}
impl<const N: usize> std::ops::IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.columns[0][i]
    }
}

impl<const N: usize, const M: usize> std::ops::Add for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, rhs: Self) -> Self::Output {
        Matrix::from_fn(|i, j| self[(i, j)] + rhs[(i, j)])
    }
}
impl<const N: usize, const M: usize> std::ops::Add<Matrix<N, M>> for Matrix<N, M> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { &self + &rhs }
}
impl<const N: usize, const M: usize> std::ops::Add<&Matrix<N, M>> for Matrix<N, M> {
    type Output = Self;
    fn add(self, rhs: &Matrix<N, M>) -> Self::Output { &self + rhs }
}
impl<const N: usize, const M: usize> std::ops::Add<Matrix<N, M>> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn add(self, rhs: Matrix<N, M>) -> Self::Output { self + &rhs }
}

impl<const N: usize, const M: usize> std::ops::AddAssign<&Matrix<N, M>> for Matrix<N, M> {
    fn add_assign(&mut self, rhs: &Matrix<N, M>) {
        (0..N).for_each(|i| (0..M).for_each(|j| self[(i, j)] += rhs[(i, j)]));
    }
}
impl<const N: usize, const M: usize> std::ops::AddAssign<Matrix<N, M>> for Matrix<N, M> {
    fn add_assign(&mut self, rhs: Matrix<N, M>) {
        *self += &rhs;
    }
}

impl<const N: usize, const M: usize> std::ops::Sub for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, rhs: Self) -> Self::Output {
        Matrix::from_fn(|i, j| self[(i, j)] - rhs[(i, j)])
    }
}
impl<const N: usize, const M: usize> std::ops::Sub<Matrix<N, M>> for Matrix<N, M> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output { &self - &rhs }
}
impl<const N: usize, const M: usize> std::ops::Sub<&Matrix<N, M>> for Matrix<N, M> {
    type Output = Self;
    fn sub(self, rhs: &Matrix<N, M>) -> Self::Output { &self - rhs }
}
impl<const N: usize, const M: usize> std::ops::Sub<Matrix<N, M>> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn sub(self, rhs: Matrix<N, M>) -> Self::Output { self - &rhs }
}

impl<const N: usize, const M: usize> std::ops::SubAssign<&Matrix<N, M>> for Matrix<N, M> {
    fn sub_assign(&mut self, rhs: &Matrix<N, M>) {
        (0..N).for_each(|i| (0..M).for_each(|j| self[(i, j)] -= rhs[(i, j)]));
    }
}
impl<const N: usize, const M: usize> std::ops::SubAssign<Matrix<N, M>> for Matrix<N, M> {
    fn sub_assign(&mut self, rhs: Matrix<N, M>) {
        *self -= &rhs;
    }
}

impl<const N: usize, const M: usize, const L: usize> std::ops::Mul<&Matrix<M, L>> for &Matrix<N, M> {
    type Output = Matrix<N, L>;
    fn mul(self, rhs: &Matrix<M, L>) -> Self::Output {
        Matrix::<N, L>::from_fn(|i, j| (0..M).map(|k| self[(i, k)] * rhs[(k, j)]).sum())
    }
}
impl<const N: usize, const M: usize, const L: usize> std::ops::Mul<Matrix<M, L>> for Matrix<N, M> {
    type Output = Matrix<N, L>;
    fn mul(self, rhs: Matrix<M, L>) -> Self::Output { &self * &rhs }
}
impl<const N: usize, const M: usize, const L: usize> std::ops::Mul<&Matrix<M, L>> for Matrix<N, M> {
    type Output = Matrix<N, L>;
    fn mul(self, rhs: &Matrix<M, L>) -> Self::Output { &self * rhs }
}
impl<const N: usize, const M: usize, const L: usize> std::ops::Mul<Matrix<M, L>> for &Matrix<N, M> {
    type Output = Matrix<N, L>;
    fn mul(self, rhs: Matrix<M, L>) -> Self::Output { self * &rhs }
}

impl<const N: usize, const M: usize> std::ops::Mul<f64> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn mul(self, rhs: f64) -> Self::Output {
        Matrix::from_fn(|i, j| self[(i, j)] * rhs)
    }
}
impl<const N: usize, const M: usize> std::ops::Mul<f64> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn mul(self, rhs: f64) -> Self::Output { &self * rhs }
}
impl<const N: usize, const M: usize> std::ops::Mul<&Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;
    fn mul(self, rhs: &Matrix<N, M>) -> Self::Output { rhs * self }
}
impl<const N: usize, const M: usize> std::ops::Mul<Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;
    fn mul(self, rhs: Matrix<N, M>) -> Self::Output { &rhs * self }
}

impl<const N: usize, const M: usize> std::ops::MulAssign<f64> for Matrix<N, M> {
    fn mul_assign(&mut self, rhs: f64) {
        (0..N).for_each(|i| (0..M).for_each(|j| self[(i, j)] *= rhs));
    }
}

impl<const N: usize, const M: usize> std::ops::Div<f64> for &Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn div(self, rhs: f64) -> Self::Output {
        Matrix::from_fn(|i, j| self[(i, j)] / rhs)
    }
}
impl<const N: usize, const M: usize> std::ops::Div<f64> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    fn div(self, rhs: f64) -> Self::Output { &self / rhs }
}

impl<const N: usize, const M: usize> std::ops::DivAssign<f64> for Matrix<N, M> {
    fn div_assign(&mut self, rhs: f64) {
        (0..N).for_each(|i| (0..M).for_each(|j| self[(i, j)] /= rhs));
    }
}
