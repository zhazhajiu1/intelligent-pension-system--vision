import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Input, Reshape
from tensorflow.keras.layers import Dense, Activation, Permute, Concatenate, Conv2D, Add, multiply, Lambda
from tensorflow.keras import backend as K

from algorithm.emotion_recognition.emotion_data_preparation import CKPlusDataset


class EmotionRecognitionModel:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)

    def build_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)

        # 卷积层
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

        # 增加通道注意力机制
        x = self.channel_attention(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def channel_attention(self, input_feature, ratio=8):
        # 获取输入特征的通道数,根据图像数据格式来确定通道轴的位置
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        # 第一个全连接层将通道数缩小到原来的 1/ratio，此处下降8倍，使用 ReLU 激活函数和 He 初始化
        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        # 第二个全连接层将通道数恢复到原始大小
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        # 对输入特征进行全局平均池化和全局最大池化操作，得到平均池化特征和最大池化特征
        # 这两个特征都被重塑为形状为 (1, 1, channel) 的张量
        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)
        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)
        # 将平均池化特征和最大池化特征传入共享的全连接层，并将它们相加,来融合不同的特征信息
        cbam_feature = Add()([avg_pool, max_pool])
        # 通过 Sigmoid 激活函数将融合后的特征映射到 [0, 1] 范围内，得到通道注意力权重
        cbam_feature = Activation('sigmoid')(cbam_feature)

        # 将输入特征与通道注意力权重相乘作为输出，增强了特征表示
        return Multiply()([input_feature, cbam_feature])

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

    model.save_model('attention_emotion_recognition_model.h5')

if __name__ == '__main__':
    main()
