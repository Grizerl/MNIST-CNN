from tensorflow import keras # type: ignore

model = keras.models.load_model("mnist_cnn_model.h5")

model.summary()
