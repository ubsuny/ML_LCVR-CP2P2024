# Use of Random Numbers in this Project

In this project, the physics is entirely classical and deterministic, so it is difficult to think of a use case for random numbers. Though they could potentially be useful in testing the model after it is created. Since the model should work on wavelengths in the range of roughly 400-100 nm, we could generate
```Python
import tensorflow as tf
random_wavelength = tf.random.uniform(shape=[1], minval=400, maxval=1000)
```
To generate a random wavelength and/or starting parameters, and see if the model can correctly reproduce the desired state. For these purposes, the "true" randomness of the numbers is unimportant.
