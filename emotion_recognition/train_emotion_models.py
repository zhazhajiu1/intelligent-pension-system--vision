import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from algorithm.emotion_recognition.emotion_data_preparation import CKPlusDataset


class EmotionRecognitionModel:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            # 将多维的特征图展平成一维
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

DATA_DIR = 'data'
def main():
    data_dir = DATA_DIR
    dataset = CKPlusDataset(data_dir)
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()

    model = EmotionRecognitionModel(input_shape=(48, 48, 1), num_classes=3) # 标签类别
    model.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=32)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    model.save_model('emotion_recognition_model.h5')

if __name__ == '__main__':
    main()
