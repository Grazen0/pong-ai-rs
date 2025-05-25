use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};

/// An n-dimensional matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<T, const R: usize> {
    data: Vec<T>,
    shape: [usize; R],
    steps: [usize; R],
}

impl<T, const R: usize> Tensor<T, R>
where
    T: Default + Clone,
{
    /// Constructs a new `Tensor<T, R>` with the given shape and filled with `T::default()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pong_ai::algebra::Tensor;
    /// let tensor = Tensor::<i32, 2>::new([2, 2]);
    /// assert_eq!(tensor[[0, 0]], 0);
    /// assert_eq!(tensor[[0, 1]], 0);
    /// assert_eq!(tensor[[1, 0]], 0);
    /// assert_eq!(tensor[[1, 1]], 0);
    /// ```
    pub fn new(shape: [usize; R]) -> Self {
        let data_len = shape.iter().product();
        Self {
            data: vec![T::default(); data_len],
            shape,
            steps: Self::calculate_steps(&shape),
        }
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: Clone,
{
    /// Fills `self` with elements by cloning `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pong_ai::algebra::Tensor;
    /// let mut tensor = Tensor::<i32, 2>::new([2, 2]);
    /// tensor.fill(1);
    /// assert_eq!(tensor[[0, 0]], 1);
    /// assert_eq!(tensor[[0, 1]], 1);
    /// assert_eq!(tensor[[1, 0]], 1);
    /// assert_eq!(tensor[[1, 1]], 1);
    /// ```
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T, const R: usize> Tensor<T, R> {
    fn calculate_steps(shape: &[usize; R]) -> [usize; R] {
        let mut steps = [0; R];
        let mut step = 1;

        for i in (0..R).rev() {
            steps[i] = step;
            step *= shape[i];
        }

        steps
    }

    /// Returns a reference to this tensor's shape.
    pub const fn shape(&self) -> &[usize; R] {
        &self.shape
    }

    /// Changes this tensor's `shape`, effectively reinterpreting its data with a new shape.
    /// Note that the product of `new_shape` must be equal to the product of the current `shape`.
    ///
    /// # Panics
    ///
    /// This method will panic if the given argument results in a different total data size.
    /// (This is: if the current and new shapes have different products).
    ///
    /// # Examples
    ///
    /// ```
    /// # use pong_ai::algebra::Tensor;
    /// let mut tensor = Tensor::<i32, 2>::new([2, 3]);
    /// tensor[[0, 0]] = 1;
    /// tensor[[0, 1]] = 2;
    /// tensor[[1, 0]] = 4;
    /// tensor[[1, 2]] = 6;
    ///
    /// tensor.reshape([3, 2]);
    /// assert_eq!(tensor[[0, 0]], 1);
    /// assert_eq!(tensor[[0, 1]], 2);
    /// assert_eq!(tensor[[1, 1]], 4);
    /// assert_eq!(tensor[[2, 1]], 6);
    /// ```
    pub fn reshape(&mut self, new_shape: [usize; R]) {
        assert_eq!(
            self.data.len(),
            new_shape.iter().product(),
            "invalid tensor reshape"
        );
        self.shape = new_shape;
        self.steps = Self::calculate_steps(&new_shape);
    }

    /// Changes this tensor's `shape`, effectively reinterpreting its data with a new shape.
    ///
    /// For a safe alternative see `reshape`.
    ///
    /// # Safety
    ///
    /// Calling this method with a `new_shape` argument that results in a different
    /// total data size may cause invalid indexing later on.
    pub unsafe fn reshape_unchecked(&mut self, new_shape: [usize; R]) {
        self.shape = new_shape;
        self.steps = Self::calculate_steps(&new_shape);
    }

    /// Returns a reference to an element of this tensor at `index` or `None` if `index` is
    /// out of bounds.
    ///
    /// See `get_mut` for a version that returns a mutable reference instead.
    pub fn get(&self, index: [usize; R]) -> Option<&T> {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        self.data.get(physical_index)
    }

    /// Returns a mutable reference to an element of this tensor at `index` or `None` if
    /// `index` is out of bounds.
    ///
    /// See `get` for a version that returns an inmutable reference instead.
    pub fn get_mut(&mut self, index: [usize; R]) -> Option<&mut T> {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        self.data.get_mut(physical_index)
    }

    /// Returns a reference to an element of this tensor at `index` without doing bounds checking.
    ///
    /// For a safe alternative see `get`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn get_unchecked(&self, index: [usize; R]) -> &T {
        unsafe {
            let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
            self.data.get_unchecked(physical_index)
        }
    }

    /// Returns a mutable reference to an element of this tensor at `index` without doing
    /// bounds checking.
    ///
    /// For a safe alternative see `get_mut`.
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is *[undefined behavior]*
    /// even if the resulting reference is not used.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; R]) -> &mut T {
        unsafe {
            let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
            self.data.get_unchecked_mut(physical_index)
        }
    }
}

impl<T, const R: usize> Index<usize> for Tensor<T, R> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const R: usize> IndexMut<usize> for Tensor<T, R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T, const R: usize> Index<[usize; R]> for Tensor<T, R> {
    type Output = T;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        &self.data[physical_index]
    }
}

impl<T, const R: usize> IndexMut<[usize; R]> for Tensor<T, R> {
    fn index_mut(&mut self, index: [usize; R]) -> &mut Self::Output {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        &mut self.data[physical_index]
    }
}

impl<T, const R: usize> Add for Tensor<T, R>
where
    T: Add<Output = T>,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "tensor dimensions do not match");

        self.data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a + b)
            .collect();
        self
    }
}

impl<T, const R: usize> AddAssign for Tensor<T, R>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape, "tensor dimensions do not match");

        for (i, el) in rhs.data.into_iter().enumerate() {
            self.data[i] += el;
        }
    }
}

impl<T, const R: usize> Sub for Tensor<T, R>
where
    T: Sub<Output = T>,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "tensor dimensions do not match");

        self.data = self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(a, b)| a - b)
            .collect();
        self
    }
}

impl<T, const R: usize> Mul for Tensor<T, R>
where
    T: Mul<Output = T>,
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        if self.shape == rhs.shape {
            self.data = self
                .data
                .into_iter()
                .zip(rhs.data)
                .map(|(a, b)| a * b)
                .collect();
            return self;
        }

        todo!("implement multiplication for tensors of different shapes");
    }
}

impl<T, const R: usize> SubAssign for Tensor<T, R>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        for (i, el) in rhs.data.into_iter().enumerate() {
            self.data[i] -= el;
        }
    }
}

impl<T> Tensor<T, 2>
where
    T: Clone,
{
    /// Returns the transpose of this 2-dimensional tensor (as in a matrix transpose).
    ///
    /// # Examples
    ///
    /// ```
    /// # use pong_ai::algebra::Tensor;
    /// let mut tensor = Tensor::<i32, 2>::new([2, 3]);
    /// tensor[[0, 0]] = 1;
    /// tensor[[0, 1]] = 2;
    /// tensor[[1, 2]] = 3;
    ///
    /// let transpose = tensor.transpose_2d();
    /// assert_eq!(transpose.shape(), &[3, 2]);
    /// assert_eq!(transpose[[0, 0]], 1);
    /// assert_eq!(transpose[[1, 0]], 2);
    /// assert_eq!(transpose[[2, 1]], 3);
    /// ```
    pub fn transpose_2d(&self) -> Self {
        let mut data = Vec::with_capacity(self.data.len());

        for j in 0..self.shape[1] {
            for i in 0..self.shape[0] {
                data.push(self[[i, j]].clone());
            }
        }

        let new_shape = [self.shape[1], self.shape[0]];
        Self {
            data,
            shape: new_shape,
            steps: Self::calculate_steps(&new_shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_indexing_linear() {
        let mut tensor = Tensor::<i32, 2>::new([2, 3]);
        tensor.data = vec![1, 2, 3, 4, 5, 6];

        assert_eq!(tensor.data[0], 1);
        assert_eq!(tensor.data[1], 2);
        assert_eq!(tensor.data[2], 3);
        assert_eq!(tensor.data[3], 4);
        assert_eq!(tensor.data[4], 5);
        assert_eq!(tensor.data[5], 6);
    }

    #[test]
    fn test_tensor_indexing_2d() {
        let mut tensor = Tensor::<i32, 2>::new([2, 3]);
        tensor.data = vec![1, 2, 3, 4, 5, 6];

        assert_eq!(tensor[[0, 0]], 1);
        assert_eq!(tensor[[0, 1]], 2);
        assert_eq!(tensor[[0, 2]], 3);
        assert_eq!(tensor[[1, 0]], 4);
        assert_eq!(tensor[[1, 1]], 5);
        assert_eq!(tensor[[1, 2]], 6);
    }

    #[test]
    fn test_tensor_indexing_3d() {
        let mut tensor = Tensor::<i32, 3>::new([3, 5, 2]);
        tensor.data[7] = 42;
        tensor.data[12] = 38;
        tensor.data[20] = 256;
        tensor.data[21] = 100;

        assert_eq!(tensor[[0, 3, 1]], 42);
        assert_eq!(tensor[[1, 1, 0]], 38);
        assert_eq!(tensor[[2, 0, 0]], 256);
        assert_eq!(tensor[[2, 0, 1]], 100);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = Tensor::<i32, 2>::new([2, 3]);
        tensor.fill(7);

        assert!(tensor.data.iter().all(|&n| n == 7));
    }

    #[test]
    fn test_tensor_add() {
        let mut tensor_a = Tensor::<i32, 2>::new([2, 3]);
        tensor_a[[0, 0]] = 7;
        tensor_a[[0, 1]] = 17;
        tensor_a[[0, 2]] = 31;
        tensor_a[[1, 0]] = 0;
        tensor_a[[1, 1]] = 63;
        tensor_a[[1, 2]] = 102;

        let mut tensor_b = Tensor::<i32, 2>::new([2, 3]);
        tensor_b[[0, 0]] = 3;
        tensor_b[[0, 1]] = 5;
        tensor_b[[0, 2]] = -2;
        tensor_b[[1, 0]] = 5;
        tensor_b[[1, 1]] = 6;
        tensor_b[[1, 2]] = 42;

        let sum = tensor_a + tensor_b;

        assert_eq!(sum[[0, 0]], 10);
        assert_eq!(sum[[0, 1]], 22);
        assert_eq!(sum[[0, 2]], 29);
        assert_eq!(sum[[1, 0]], 5);
        assert_eq!(sum[[1, 1]], 69);
        assert_eq!(sum[[1, 2]], 144);
    }

    #[test]
    fn test_tensor_add_assign() {
        let mut tensor_a = Tensor::<i32, 2>::new([2, 3]);
        tensor_a[[0, 0]] = 7;
        tensor_a[[0, 1]] = 17;
        tensor_a[[0, 2]] = 31;
        tensor_a[[1, 0]] = 0;
        tensor_a[[1, 1]] = 63;
        tensor_a[[1, 2]] = 102;

        let mut tensor_b = Tensor::<i32, 2>::new([2, 3]);
        tensor_b[[0, 0]] = 3;
        tensor_b[[0, 1]] = 5;
        tensor_b[[0, 2]] = -2;
        tensor_b[[1, 0]] = 5;
        tensor_b[[1, 1]] = 6;
        tensor_b[[1, 2]] = 42;

        tensor_a += tensor_b;

        assert_eq!(tensor_a[[0, 0]], 10);
        assert_eq!(tensor_a[[0, 1]], 22);
        assert_eq!(tensor_a[[0, 2]], 29);
        assert_eq!(tensor_a[[1, 0]], 5);
        assert_eq!(tensor_a[[1, 1]], 69);
        assert_eq!(tensor_a[[1, 2]], 144);
    }

    #[test]
    fn test_tensor_sub() {
        let mut tensor_a = Tensor::<i32, 2>::new([2, 3]);
        tensor_a[[0, 0]] = 7;
        tensor_a[[0, 1]] = 17;
        tensor_a[[0, 2]] = 31;
        tensor_a[[1, 0]] = 0;
        tensor_a[[1, 1]] = 63;
        tensor_a[[1, 2]] = 102;

        let mut tensor_b = Tensor::<i32, 2>::new([2, 3]);
        tensor_b[[0, 0]] = 3;
        tensor_b[[0, 1]] = 5;
        tensor_b[[0, 2]] = -2;
        tensor_b[[1, 0]] = 5;
        tensor_b[[1, 1]] = 6;
        tensor_b[[1, 2]] = 42;

        let diff = tensor_a - tensor_b;

        assert_eq!(diff[[0, 0]], 4);
        assert_eq!(diff[[0, 1]], 12);
        assert_eq!(diff[[0, 2]], 33);
        assert_eq!(diff[[1, 0]], -5);
        assert_eq!(diff[[1, 1]], 57);
        assert_eq!(diff[[1, 2]], 60);
    }

    #[test]
    fn test_tensor_sub_assign() {
        let mut tensor_a = Tensor::<i32, 2>::new([2, 3]);
        tensor_a[[0, 0]] = 7;
        tensor_a[[0, 1]] = 17;
        tensor_a[[0, 2]] = 31;
        tensor_a[[1, 0]] = 0;
        tensor_a[[1, 1]] = 63;
        tensor_a[[1, 2]] = 102;

        let mut tensor_b = Tensor::<i32, 2>::new([2, 3]);
        tensor_b[[0, 0]] = 3;
        tensor_b[[0, 1]] = 5;
        tensor_b[[0, 2]] = -2;
        tensor_b[[1, 0]] = 5;
        tensor_b[[1, 1]] = 6;
        tensor_b[[1, 2]] = 42;

        tensor_a -= tensor_b;

        assert_eq!(tensor_a[[0, 0]], 4);
        assert_eq!(tensor_a[[0, 1]], 12);
        assert_eq!(tensor_a[[0, 2]], 33);
        assert_eq!(tensor_a[[1, 0]], -5);
        assert_eq!(tensor_a[[1, 1]], 57);
        assert_eq!(tensor_a[[1, 2]], 60);
    }

    #[test]
    fn test_tensor_transpose() {
        let mut mat = Tensor::<i32, 2>::new([3, 2]);
        mat[[0, 0]] = 1;
        mat[[0, 1]] = 2;
        mat[[1, 0]] = 3;
        mat[[1, 1]] = 4;
        mat[[2, 0]] = 5;
        mat[[2, 1]] = 6;

        let mat_t = mat.transpose_2d();

        assert_eq!(mat_t.shape(), &[2, 3]);
        assert_eq!(mat_t[[0, 0]], 1);
        assert_eq!(mat_t[[0, 1]], 3);
        assert_eq!(mat_t[[0, 2]], 5);
        assert_eq!(mat_t[[1, 0]], 2);
        assert_eq!(mat_t[[1, 1]], 4);
        assert_eq!(mat_t[[1, 2]], 6);
    }
}
