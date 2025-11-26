# src/main.py
import numpy as np # type: ignore
from tensorflow import keras # type: ignore
from model import build_cnn

def load_data():
    # Завантажуємо MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Нормалізація
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Додаємо канал (28x28x1)
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_data()

    model = build_cnn()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Навчання
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    # Збереження моделі
    model.save("mnist_cnn_model.h5")

    # Оцінка на тесті
    loss, acc = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
