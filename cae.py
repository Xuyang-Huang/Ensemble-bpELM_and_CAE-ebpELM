#-- coding: utf-8 --
#@Time : 2021/5/16 23:31
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@Software: PyCharm

import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
import random
from tensorflow.keras.layers import Conv1D,Activation,MaxPooling1D,Conv1DTranspose,Dropout,BatchNormalization,Input,AveragePooling1D,Flatten,Dense
from tensorflow.keras import Sequential


class CAE:
    """Convolutional Auto-Encoder.

    Attributes:
        interval_dict: A dict including the IoIs of corresponding subkeys.
    """
    def __init__(self, interval_dict):
        self.interval_dict = interval_dict
        input_length = interval_dict[0][1] - interval_dict[0][0]
        self.new_input_length = int(2 ** np.ceil(np.log2(input_length)))
        self.encoder = None
        print('Input length is changed to', self.new_input_length, 'for CAE training.')

    def random_select(self, data):
        byte_idx = random.randint(0, 15)
        raw_length = self.interval_dict[byte_idx][1] - self.interval_dict[byte_idx][0]
        start_point = self.interval_dict[byte_idx][0] - (self.new_input_length - raw_length)//2
        end_point = self.interval_dict[byte_idx][1] + (self.new_input_length - raw_length) - (self.new_input_length - raw_length)//2
        data = data[start_point: end_point, tf.newaxis]
        # data = (data - self.min)/(self.max - self.min)
        return (data, data)

    def preprocess(sefl, data, batch_size):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.map(sefl.random_select)
        ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        return ds

    @staticmethod
    def build_encoder(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        def seq(filters, kernel, pool):
            sub_model = Sequential([
                Conv1D(filters, kernel, strides=1, activation=None, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(pool, pool, padding='same')])
            return sub_model
        x = seq(16, 11, 4)(inputs)
        x = seq(32, 7, 4)(x)
        x = seq(64, 5, 4)(x)
        x = seq(128, 3, 2)(x)
        x = seq(128, 3, 2)(x)
        x = seq(256, 3, 2)(x)
        outputs = Conv1D(256, 3, strides=1, activation='sigmoid', padding='same')(x)
        model = tf.keras.Model(inputs, outputs, name='encoder')
        model.summary()
        return model

    @staticmethod
    def build_decoder(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        def seq(filters, kernel, stride):
            sub_model = Sequential([
                Conv1DTranspose(filters, kernel, stride, activation=None, padding='same'),
                BatchNormalization(),
                Activation(tf.nn.leaky_relu)])
            return sub_model
        x = Conv1DTranspose(256, 3, 1, activation=tf.nn.leaky_relu, padding='same')(inputs)
        x = seq(256, 3, 2)(x)
        x = seq(128, 3, 2)(x)
        x = seq(128, 3, 2)(x)
        x = seq(64, 5, 4)(x)
        x = seq(32, 7, 4)(x)
        x = Conv1DTranspose(16, 11, 4, activation=tf.nn.leaky_relu, padding='same')(x)
        outputs = Conv1DTranspose(1, 3, 1, activation=None, padding='same')(x)
        model = tf.keras.Model(inputs, outputs, name='decoder')
        model.summary()
        return model

    def auto_encoder(self, length):
        input_shape = (length, 1)
        num_2_strides_pooling = 0
        num_4_strides_pooling = 0

        tmp_length = length
        while tmp_length > 8:
            tmp_length /= 4
            num_4_strides_pooling += 1

        while tmp_length > 4:
            tmp_length /= 2
            num_2_strides_pooling += 1

        encoder = self.build_encoder(input_shape)
        decoder = self.build_decoder(encoder.output_shape[1:])
        inputs = Input(shape=input_shape)
        x = encoder(inputs)
        outputs = decoder(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self):
        model = self.auto_encoder(self.new_input_length)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=tf.keras.metrics.MeanAbsoluteError())
        return model

    def preprocess_ds(self, data, batch_size):
        train_data = data[:int(len(data) * 0.9)]
        val_data = data[int(len(data) * 0.9):]
        train_ds = self.preprocess(train_data, batch_size)
        val_ds = self.preprocess(val_data, batch_size)
        return train_ds, val_ds

    def train(self, input_data, model_save_path, epochs=200, batch_size=512):
        np.random.shuffle(input_data)
        train_ds, val_ds = self.preprocess_ds(input_data, batch_size)
        model = self.get_model()
        callbacks = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', verbose=0,
                                                       save_best_only=True, save_weights_only=False, mode='auto',
                                                       save_freq='epoch', options=None)

        model.fit(x=train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    def encoder_inference(self, model_path):
        model = tf.keras.models.load_model(model_path)
        encoder = keras.Model(inputs=model.get_layer('encoder').input, outputs=model.get_layer('encoder').output)
        return encoder

    def encoder_preprocessing(self, data, model_path, i_byte):
        if not self.encoder:
            self.encoder = self.encoder_inference(model_path)
        raw_length = self.interval_dict[i_byte][1] - self.interval_dict[i_byte][0]
        start_point = self.interval_dict[i_byte][0] - (self.new_input_length - raw_length) // 2
        end_point = self.interval_dict[i_byte][1] + (self.new_input_length - raw_length) - (
                    self.new_input_length - raw_length) // 2
        data = data[:, start_point:end_point]
        output_data = self.encoder.predict(data)
        return output_data








