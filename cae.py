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
    """Convolutional Auto-Encoder. Layer number will automatically adjust according to input length.

    Attributes:
        interval_dict: A dict including the IoIs of corresponding subkeys.
        max: max value of traces, for data rescaling.
        min: min value of traces, for data rescaling.
    """
    def __init__(self, interval_dict, max, min):
        self.interval_dict = interval_dict
        self.max = max
        self.min = min
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
        data = (data - self.min)/(self.max - self.min)
        return (data, data)

    def preprocess(sefl, data, batch_size):
        AUTOTUNE = tf.data.AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.shuffle(50000, reshuffle_each_iteration=True)
        ds = ds.map(sefl.random_select)
        ds = ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        return ds

    def build_encoder(self, input_shape, num_2_strides_pooling, num_4_strides_pooling):
        inputs = tf.keras.Input(shape=input_shape)
        filters_lst = [4]
        x = Sequential([
            Conv1D(4, 11, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu')])(inputs)
        for i in range(num_4_strides_pooling):
            if (x.shape[1] * x.shape[2] / 4) < (self.new_input_length / 2):
                filters = int((self.new_input_length / 2) / (x.shape[1] / 4))
            else:
                filters = 4
            filters_lst.append(filters)
            x = Sequential([
                Conv1D(filters, 11, strides=1, activation=None, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(4, 4, padding='same')])(x)

        for i in range(num_2_strides_pooling - 1):
            if (x.shape[1] * x.shape[2] / 2) < (self.new_input_length / 2):
                filters = int((self.new_input_length / 2) / (x.shape[1] / 2))
            else:
                filters = 4
            filters_lst.append(filters)
            x = Sequential([
                Conv1D(filters, 5, strides=1, activation=None, padding='same'),
                BatchNormalization(),
                Activation('relu'),
                MaxPooling1D(2, 2, padding='same')])(x)

        if (x.shape[1] * x.shape[2] / 2) < (self.new_input_length / 2):
            filters = int((self.new_input_length / 2) / (x.shape[1] / 2))
        else:
            filters = 4
        filters_lst.append(filters)
        outputs = Sequential([
            Conv1D(filters, 5, strides=1, activation=None, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2, 2, padding='same')])(x)
        model = tf.keras.Model(inputs, outputs, name='encoder')
        model.summary()
        return model, filters_lst

    @staticmethod
    def build_decoder(input_shape, filters_lst, num_2_strides_pooling, num_4_strides_pooling):
        inputs = tf.keras.Input(shape=input_shape)
        x = Sequential([
            Conv1DTranspose(filters_lst.pop(), 5, 2, activation=None, padding='same'),
            BatchNormalization(),
            Activation(tf.nn.leaky_relu)])(inputs)
        for i in range(num_2_strides_pooling - 1):
            x = Sequential([
                Conv1DTranspose(filters_lst.pop(), 5, 2, activation=None, padding='same'),
                BatchNormalization(),
                Activation(tf.nn.leaky_relu)])(x)

        for i in range(num_4_strides_pooling):
            x = Sequential([
                    Conv1DTranspose(filters_lst.pop(), 11, 4, activation=None, padding='same'),
                    BatchNormalization(),
                    Activation(tf.nn.leaky_relu)])(x)
        outputs = Conv1DTranspose(1, 3, 1, activation=None, padding='same')(x)
        model = tf.keras.Model(inputs, outputs, name='decoder')
        model.summary()
        return model

    def auto_encoder(self, length):
        input_shape = (length, 1)
        num_2_strides_pooling = 0
        num_4_strides_pooling = 0

        tmp_length = length
        while tmp_length > 64:
            tmp_length /= 4
            num_4_strides_pooling += 1

        while tmp_length > 4:
            tmp_length /= 2
            num_2_strides_pooling += 1

        encoder, filters_lst = self.build_encoder(input_shape, num_2_strides_pooling, num_4_strides_pooling)
        decoder = self.build_decoder(encoder.output_shape[1:], filters_lst, num_2_strides_pooling, num_4_strides_pooling)
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
                                                       save_best_only=True, save_weights_only=True, mode='auto',
                                                       save_freq='epoch', options=None)

        model.fit(x=train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

    def encoder_inference(self, model_path):
        model = self.auto_encoder(self.new_input_length)
        model.load_weights(model_path)
        encoder = keras.Model(inputs=model.get_layer('encoder').input, outputs=model.get_layer('encoder').output)
        return encoder

    def encoder_preprocessing(self, data, model_path, i_byte):
        if not self.encoder:
            self.encoder = self.encoder_inference(model_path)
        raw_length = self.interval_dict[i_byte][1] - self.interval_dict[i_byte][0]
        start_point = self.interval_dict[i_byte][0] - (self.new_input_length - raw_length) // 2
        end_point = self.interval_dict[i_byte][1] + (self.new_input_length - raw_length) - (
                    self.new_input_length - raw_length) // 2

        output_data = self.encoder.predict(self.rescale(data[:, start_point:end_point]))

        return output_data

    def rescale(self, data):
        return (data - self.min) / (self.max - self.min)






