# mixed-sfs

- Unsupervised spectral feature selection for mixed data (numerical and categorical). 
- GPU-accelerated using PyCUDA. 

**Developed using the paper**: Solorio-Fernández, S., Martínez-Trinidad, J.F. and Carrasco-Ochoa, J.A., "A new unsupervised spectral feature selection method for mixed data: a filter approach". *Pattern Recognition*, vol. 72, pp.314-326, 2017. (https://doi.org/10.1016/j.patcog.2017.07.020)

---

## Dependencies

1. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. [pycuda](https://pypi.org/project/pycuda/)
3. [scikit-cuda](https://pypi.org/project/scikit-cuda/)
4. [numpy](https://numpy.org/)

## Instructions

1. Ensure all dependencies are met. 
2. `git clone` the repository and use directly. No packaging done yet, sorry.
3. To perform spectral feature selection on mixed data:

```
import mixed-sfs.spectralFeatureSelection as sfs
featureWeights = sfs.selectFeatures(inData, maxNumClusters, catCols)
```

Refer in-code documentation for more details on the function `selectFeatures`. 

## Testing

Code was developed and tested on:
- Ubuntu 20.04.1
- python version 3.8.2
- numpy version 1.19.1
- CUDA toolkit version 11.0
- pycuda version 2019.1.2
- scikit-cuda version 0.5.3
