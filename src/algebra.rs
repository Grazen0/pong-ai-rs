use std::{
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    slice::{Iter, IterMut},
    vec::IntoIter,
};

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
    pub fn new(shape: [usize; R]) -> Self {
        let data_len = shape.iter().product();
        Self {
            data: vec![T::default(); data_len],
            shape,
            steps: Self::calculate_steps(&shape),
        }
    }

    pub fn reshape(&mut self, new_shape: [usize; R]) {
        self.shape = new_shape;
        self.steps = Self::calculate_steps(&new_shape);
        self.data.resize(new_shape.iter().product(), T::default());
    }
}

impl<T, const R: usize> Tensor<T, R>
where
    T: Clone,
{
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    fn broadcast<F>(mut self, rhs: Self, f: F) -> Self
    where
        F: Fn((T, T)) -> T,
    {
        if self.shape == rhs.shape {
            self.data = self.data.into_iter().zip(rhs.data).map(f).collect();
            return self;
        }

        todo!()
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

    pub const fn shape(&self) -> &[usize; R] {
        &self.shape
    }

    pub fn get(&self, index: [usize; R]) -> Option<&T> {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        self.data.get(physical_index)
    }

    pub fn get_mut(&mut self, index: [usize; R]) -> Option<&mut T> {
        let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
        self.data.get_mut(physical_index)
    }

    pub unsafe fn get_unchecked(&self, index: [usize; R]) -> &T {
        unsafe {
            let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
            self.data.get_unchecked(physical_index)
        }
    }

    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; R]) -> &mut T {
        unsafe {
            let physical_index: usize = index.iter().zip(self.steps).map(|(i, s)| i * s).sum();
            self.data.get_unchecked_mut(physical_index)
        }
    }

    pub fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
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
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.broadcast(rhs, |(a, b)| a + b)
    }
}

impl<T, const R: usize> Add<T> for Tensor<T, R>
where
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(mut self, rhs: T) -> Self::Output {
        self.data = self.data.into_iter().map(|x| x + rhs.clone()).collect();
        self
    }
}

impl<T, const R: usize> Neg for Tensor<T, R>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.data = self.data.into_iter().map(T::neg).collect();
        self
    }
}

impl<T, const R: usize> Sub for Tensor<T, R>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.broadcast(rhs, |(a, b)| a - b)
    }
}

impl<T, const R: usize> Sub<T> for Tensor<T, R>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.data = self.data.into_iter().map(|x| x - rhs.clone()).collect();
        self
    }
}

impl<T, const R: usize> Mul for Tensor<T, R>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.broadcast(rhs, |(a, b)| a * b)
    }
}

impl<T, const R: usize> Mul<T> for Tensor<T, R>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.data = self.data.into_iter().map(|x| x * rhs.clone()).collect();
        self
    }
}

impl<T, const R: usize> Div for Tensor<T, R>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.broadcast(rhs, |(a, b)| a / b)
    }
}

impl<T, const R: usize> Div<T> for Tensor<T, R>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(mut self, rhs: T) -> Self::Output {
        self.data = self.data.into_iter().map(|x| x / rhs.clone()).collect();
        self
    }
}

impl<T, const R: usize> IntoIterator for Tensor<T, R> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_indexing_linear() {
        let mut tensor = Tensor::<i32, 2>::new([2, 3]);
        tensor.data = vec![1, 2, 3, 4, 5, 6];

        assert_eq!(tensor[0], 1);
        assert_eq!(tensor[1], 2);
        assert_eq!(tensor[2], 3);
        assert_eq!(tensor[3], 4);
        assert_eq!(tensor[4], 5);
        assert_eq!(tensor[5], 6);
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
    fn test_tensor_scalar_mul() {
        let mut tensor_a = Tensor::<i32, 2>::new([2, 3]);
        tensor_a[[0, 0]] = 7;
        tensor_a[[0, 2]] = 31;
        tensor_a[[1, 0]] = 0;
        tensor_a[[1, 2]] = 102;

        let tensor_b = tensor_a * 3;
        assert_eq!(tensor_b[[0, 0]], 21);
        assert_eq!(tensor_b[[0, 2]], 93);
        assert_eq!(tensor_b[[1, 0]], 0);
        assert_eq!(tensor_b[[1, 2]], 306);
    }
}
