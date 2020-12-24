# SAR-Wave-Height

Predicting significant wave height from synthetic aperture radar (SAR) using the method described in Quach, et. al. 2020, [*Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar*](https://ieeexplore.ieee.org/document/9143500). Also available [*here*](https://authors.library.caltech.edu/104562/1/09143500.pdf)

### Quick start:
1. *scripts/create_dataset_from_nc.ipynb* processes a netcdf file into a dataset for training or making predictions.
1. *notebooks/train_model_heteroskedastic.ipynb* trains a model with uncertainty predictions (heteroskedastic regression).
1. *notebooks/predict.ipynb* loads a model and makes predictions.

### Citation:
```
@article{quach2020deep,
  author={B. {Quach} and Y. {Glaser} and J. E. {Stopa} and A. A. {Mouche} and P. {Sadowski}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar}, 
  year={2020},
  volume={},
  number={},
  pages={1-9},
  doi={10.1109/TGRS.2020.3003839}
}
```
