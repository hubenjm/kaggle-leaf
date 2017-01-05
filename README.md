# kaggle-leaf
Predicting the species of plant from leaf image data

**Includes various methods for fitting the leaf data to the proper species name**
  * Random Forest Classifier using just the provided derived numerical features from kaggle
  * Convolutional Neural Network using both the images themselves and the provided derived numerical features from kaggle
  * (future) Utilize a more sophisticated, low-level image preprocessing technique to extract the leaf boundary curve from each image with denoising and compile into a standard 1D vector of $r$-values with respect to polar coordinate $\theta$. Then combine this data with the given numerical features from kaggle in a convolutional neural network similar to the previous approach.
