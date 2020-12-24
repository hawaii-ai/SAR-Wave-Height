# SAR-Wave-Height

Predicting significant wave height from synthetic aperture radar (SAR) using the method described in Quach, et. al. 2020, [*Deep Learning for Predicting Significant Wave Height From Synthetic Aperture Radar*](https://ieeexplore.ieee.org/document/9143500). Also available [*here*](https://authors.library.caltech.edu/104562/1/09143500.pdf)

Quikstart:
-*scripts/create_dataset_from_nc.ipynb* processes a netcdf5 file into a dataset for training or making predictions.
-*notebooks/train_model_heteroskedastic.ipynb* trains a model with uncertainty predictions (heteroskedastic regression).
-*notebooks/predict.ipynb* loads a model and makes predictions.


