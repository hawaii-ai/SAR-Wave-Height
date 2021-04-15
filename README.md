# SAR-Wave-Height

Predicting significant wave height from synthetic aperture radar (SAR) using the method described in Quach, et. al. 2020, [*Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar*](https://ieeexplore.ieee.org/document/9143500). Also available [*here*](https://authors.library.caltech.edu/104562/1/09143500.pdf)

### Quick start:
1. Process a netcdf file into a dataset for training or making predictions: *scripts/create_dataset_from_nc.ipynb*
1. Train a model with uncertainty predictions (heteroskedastic regression): *notebooks/train_model_heteroskedastic.ipynb* 
1. Load a model and make predictions: *notebooks/predict.ipynb*

### Citation:
```
@article{quach2020deep,
  author={B. {Quach} and Y. {Glaser} and J. E. {Stopa} and A. A. {Mouche} and P. {Sadowski}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar}, 
  year={2021},
  volume={59},
  number={3},
  pages={1859-1867},
  doi={10.1109/TGRS.2020.3003839}}
```
