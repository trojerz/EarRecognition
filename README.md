### Overview

This code for this project was created by Žiga Trojer. Simple CNN architecture was used for ear recognition based on 1000 images of 100 different persons. We tackeled gender and ethnicity recognition.


### Dependencies

* Python 3.6
* Numpy
* TensorFlow 2.1.0
* Keras 2.3.1
* OpenCV

Exported environment: environment.yml

### Instructions

 * Install all dependencies from the `environment.yml` file (I did not use Jupyter Notebook due to issues with incompatibility of Tensorflow and Jupyter Notebook).
 * Put AWE dataset in `AWEForSegmentation` folder.
 * Run the `preprocess_data.py` to get cropped and resized images of ears and the right annotations. All processed pictures will be in a `train` and `test` folder, annotations fill be in a main folder.
 * Run `python train_model.py gender` for training the gender model or `python train_model.py ethnicity` for training ethnicity model.
    + Model for gender prediction is in `gender_model.py` and for ethnicity prediction is in `ethnicity_model.py`.
    + Metrics accuracy and confusion matrix are given.
 * For predicting, run `python predict.py gender 1` to predict gender on image `0001.png` in file `test_predict`, so make sure to put images in that file with the right names.

### Models

All models and results are saved in a `gender_models` and `ethnicity_models` file.