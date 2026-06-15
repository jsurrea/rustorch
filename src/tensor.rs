#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides_from_shape(shape)?;
        let expected = shape.iter().product();
        if data.len() != expected {
            return Err(TensorError::NumelMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
        })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let strides = Self::compute_strides_from_shape(shape)?;
        Ok(Self {
            data: vec![0.0; shape.iter().product()],
            shape: shape.to_vec(),
            strides,
        })
    }

    fn compute_strides_from_shape(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
        if shape.is_empty() {
            return Err(TensorError::InvalidShape {
                shape: shape.to_vec(),
            });
        }
        if shape.iter().any(|&dim| dim == 0) {
            return Err(TensorError::InvalidShape {
                shape: shape.to_vec(),
            });
        }
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        Ok(strides)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn index_to_offset(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.ndim() {
            return Err(TensorError::IncompatibleShapes {
                expected: self.shape.clone(),
                got: index.to_vec(),
            });
        }
        let mut offset = 0;
        for i in 0..index.len() {
            if index[i] >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds {
                    index: index.to_vec(),
                    shape: self.shape.clone(),
                });
            }
            offset += index[i] * self.strides[i];
        }
        Ok(offset)
    }

    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let offset = self.index_to_offset(index)?;
        Ok(self.data[offset])
    }

    pub fn set(&mut self, index: &[usize], value: f32) -> Result<(), TensorError> {
        let offset = self.index_to_offset(index)?;
        self.data[offset] = value;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum TensorError {
    NumelMismatch {
        expected: usize,
        got: usize,
    },
    InvalidShape {
        shape: Vec<usize>,
    },
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    IncompatibleShapes {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidDimension {
        dim: i32,
        ndim: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_ok() {
        let t = Tensor::from_vec(vec![1., 2., 3., 4.], &[2, 2]).unwrap();
        assert_eq!(t.numel(), 4);
    }
    #[test]
    fn from_vec_wrong_length() {
        let err = Tensor::from_vec(vec![1., 2.], &[2, 2]).unwrap_err();
        assert_eq!(
            err,
            TensorError::NumelMismatch {
                expected: 4,
                got: 2
            }
        );
    }
    #[test]
    fn zeros_has_correct_shape_and_values() {
        let t = Tensor::zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn index_to_offset_row_major() {
        let t = Tensor::zeros(&[2, 3]).unwrap();
        assert_eq!(t.index_to_offset(&[0, 2]).unwrap(), 2);
        assert_eq!(t.index_to_offset(&[1, 0]).unwrap(), 3);
        assert_eq!(t.index_to_offset(&[1, 2]).unwrap(), 5);
    }
    #[test]
    fn index_to_offset_out_of_bounds() {
        let t = Tensor::zeros(&[2, 3]).unwrap();
        let err = t.index_to_offset(&[2, 0]).unwrap_err();
        assert_eq!(
            err,
            TensorError::IndexOutOfBounds {
                index: vec![2, 0],
                shape: vec![2, 3],
            }
        );
    }
    #[test]
    fn index_to_offset_incompatible_shapes() {
        let t = Tensor::zeros(&[2, 3]).unwrap();
        let err = t.index_to_offset(&[0, 0, 0]).unwrap_err();
        assert_eq!(
            err,
            TensorError::IncompatibleShapes {
                expected: vec![2, 3],
                got: vec![0, 0, 0],
            }
        );
    }
    #[test]
    fn compute_strides_3d_row_major() {
        assert_eq!(
            Tensor::compute_strides_from_shape(&[2, 3, 4]).unwrap(),
            vec![12, 4, 1]
        );
    }

    #[test]
    fn compute_strides_rejects_invalid_shape() {
        assert_eq!(
            Tensor::compute_strides_from_shape(&[]).unwrap_err(),
            TensorError::InvalidShape { shape: vec![] }
        );
        assert_eq!(
            Tensor::compute_strides_from_shape(&[2, 0, 4]).unwrap_err(),
            TensorError::InvalidShape {
                shape: vec![2, 0, 4]
            }
        );
    }

    #[test]
    fn get_set_round_trip_2d() {
        let mut t = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();

        assert_eq!(t.get(&[0, 0]).unwrap(), 1.);
        assert_eq!(t.get(&[0, 2]).unwrap(), 3.);
        assert_eq!(t.get(&[1, 1]).unwrap(), 5.);

        t.set(&[1, 1], 99.).unwrap();
        assert_eq!(t.get(&[1, 1]).unwrap(), 99.);

        assert_eq!(t.get(&[0, 1]).unwrap(), 2.);

        assert_eq!(t.data()[4], 99.);
    }
}
