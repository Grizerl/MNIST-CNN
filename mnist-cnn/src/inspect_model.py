from tensorflow import keras # type: ignore

# Завантажуємо модель
model = keras.models.load_model("mnist_cnn_model.h5")

# Виводимо архітектуру
model.summary()
