#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@Software: PyCharm

import numpy as np


class RVRR:
    """A random vector based ridge regression classifier with or without slight BP. e.g., RVFL and ELM.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        n_class: An integer of number of class. When n_class is 1, a binary model will be build.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        weights_initialization: Weight initialization method, uniform or he initialization.
        links: A bool number, whether use a direct link. RVFL needs the direct link, ELM does not.
        use_bias: A bool number, whether use a direct link. RVFL needs the direct link, ELM does not.
        hidden_layer_norm: A bool number, whether use a normalization on the output of hidden layer.
    """
    def __init__(self, n_nodes, lam, n_class,  w_random_vec_range, b_random_vec_range, activation, same_feature=False
                 , weights_initialization='uniform', links=True, use_bias=True, hidden_layer_norm=False):
        assert weights_initialization in ['uniform', 'he'], 'uniform or he should be used as weights_initialization!'

        self.n_nodes = n_nodes
        self.lam = lam
        self.n_class = n_class
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)

        a_g = ActivationGradient()
        self.activation_gradient = getattr(a_g, activation+'_gradient')

        self.data_std = None
        self.data_mean = None
        self.hidden_layer_std = None
        self.hidden_layer_mean = None
        self.same_feature = same_feature
        self.weights_init = weights_initialization
        self.links = links
        self.use_bias = use_bias
        self.hidden_layer_norm = hidden_layer_norm

    def train_once(self, data, label, seed=None):
        """

        :param data: Training data.
        :param label: Training label.
        :return: No return
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'

        data = self.standardize(data, True)  # Normalization data
        n_feature = len(data[0])

        if self.random_weights is None:
            self.random_weights = self.get_random_weights(n_feature, self.n_nodes, self.w_random_range)
        if self.random_bias is None:
            self.random_bias = self.get_random_bias(1, self.n_nodes, self.b_random_range)

        if self.use_bias:
            h = np.dot(data, self.random_weights) + self.random_bias
        else:
            h = np.dot(data, self.random_weights)

        a = self.activation_function(h)

        if self.links:
            d = np.concatenate([a, data], axis=1)
        else:
            d = a
        if self.hidden_layer_norm:
            d = self.standardize_hidden_layer(d, True)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        if len(label.shape) > 1:
            y = label
        else:
            y = self.one_hot(label, self.n_class)
        if d.shape[0] > d.shape[1]:
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(d.shape[0]) + np.dot(d, d.T))).dot(y)

    def train(self, data, label, epochs=0, lr=None, cp=0, seed=None):
        """ Training function with or without slight BP

        :param data: Training data.
        :param label: Training label.
        :param epochs: An integer of training epochs when BP is needed. Set to 0 for no BP training.
        :param lr: A list of two integers, start of lr and end of lr for scheduled learning rate.
            Set to same values for steady learning rate.
        :param cp: An integer of scheduled learning rate factor.
        :param seed: Random seed.
        :return: No return
        """

        if seed:
            np.random.seed(seed)
        if len(label.shape) <= 1:
            label = self.one_hot(label, self.n_class)

        _ = self.standardize(data, True)

        for epoch in range(epochs):
            self.train_once(data, label)
            prediction, tmp_h = self.predict_for_bp(data)
            if self.hidden_layer_norm:
                tmp_gradient = self.activation_gradient(np.dot(2 * (prediction - label), self.beta[:self.n_nodes].T) * 1/self.hidden_layer_std, tmp_h)
            else:
                tmp_gradient = self.activation_gradient(np.dot(2 * (prediction - label), self.beta[:self.n_nodes].T), tmp_h)
            gradient_w = np.dot(self.standardize(data, False).T, tmp_gradient)
            curr_lr = (lr[1] + (lr[0] - lr[1]) * (1 - (epoch / epochs)) ** cp)
            gradient_w = gradient_w * curr_lr
            self.random_weights -= gradient_w
            if self.use_bias:
                gradient_b = tmp_gradient.sum(axis=0).reshape([1, -1])
                self.random_bias -= curr_lr * gradient_b
        self.train_once(data, label)
        if seed:
            np.random.seed(None)

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """
        data = self.standardize(data, False)  # Normalization data

        if self.use_bias:
            h = np.dot(data, self.random_weights) + self.random_bias
        else:
            h = np.dot(data, self.random_weights)

        a = self.activation_function(h)
        if self.links:
            d = np.concatenate([a, data], axis=1)
        else:
            d = a
        if self.hidden_layer_norm:
            d = self.standardize_hidden_layer(d, False)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        result = self.softmax(np.dot(d, self.beta))

        if not output_prob:
            if self.n_class > 1:
                result = np.argmax(result, axis=1)
            else:
                tmp_result = np.arrayy(result)
                result = np.ones_like(tmp_result)
                result[np.where(tmp_result < 0)] = -1
        return result

    def predict_for_bp(self, data):
        """
        :param data: Predict data.
        :return: Inference result and intermediate variables h.
        """
        data = self.standardize(data, False)  # Normalization data

        if self.use_bias:
            h = np.dot(data, self.random_weights) + self.random_bias
        else:
            h = np.dot(data, self.random_weights)

        a = self.activation_function(h)
        if self.links:
            d = np.concatenate([a, data], axis=1)
        else:
            d = a
        if self.hidden_layer_norm:
            d = self.standardize_hidden_layer(d, False)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)


        result = np.dot(d, self.beta)
        return result, h

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: Accuracy.
        """

        assert len(data.shape) > 1, 'Data shape should be [n, dim].'
        assert len(data) == len(label), 'Label number does not match data number.'
        assert len(label.shape) == 1, 'Label should be 1-D array.'

        data = self.standardize(data, False)  # Normalization data

        if self.use_bias:
            h = np.dot(data, self.random_weights) + self.random_bias
        else:
            h = np.dot(data, self.random_weights)

        a = self.activation_function(h)
        if self.links:
            d = np.concatenate([a, data], axis=1)
        else:
            d = a
        if self.hidden_layer_norm:
            d = self.standardize_hidden_layer(d, False)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        if self.n_class > 1:
            result = self.softmax(np.dot(d, self.beta))
            result = np.argmax(result, axis=1)
        else:
            tmp_result = np.dot(d, self.beta).reshape([-1])
            result = np.ones_like(tmp_result)
            result[np.where(tmp_result < 0)] = -1

        acc = np.sum(np.equal(result, label))/len(label)
        return acc

    def get_random_weights(self, m, n, scale_range):
        if self.weights_init == 'uniform':
            x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        elif self.weights_init == 'he':
            x = np.random.random([m, n]) * np.sqrt(2/(m*n))
        return x

    def get_random_bias(self, m, n, scale_range):
        if self.weights_init == 'uniform':
            x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        elif self.weights_init == 'he':
            x = np.random.random([m, n]) * 0
        return x

    def one_hot(self, x, n_class):
        if n_class == 1:
            return x[:, np.newaxis]
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x, update):
        if self.same_feature is True:
            if update:
                self.data_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if update:
                self.data_mean = np.mean(x)
            return (x - self.data_mean) / self.data_std
        else:
            if update:
                self.data_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if update:
                self.data_mean = np.mean(x, axis=0)
            return (x - self.data_mean) / self.data_std

    def standardize_hidden_layer(self, x, update):
        if self.same_feature is True:
            if update:
                self.hidden_layer_std = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if update:
                self.hidden_layer_mean = np.mean(x)
            return (x - self.hidden_layer_mean) / self.hidden_layer_std
        else:
            if update:
                self.hidden_layer_std = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if update:
                self.hidden_layer_mean = np.mean(x, axis=0)
            return (x - self.hidden_layer_mean) / self.hidden_layer_std

    def softmax(self, x):
        x -= np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
        return x

    def normalize(self, x):
        return (x - np.min(x))/(np.max(x) - np.min(x))


class Activation:
    def tanh(self, x):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def sine(self, x):
        return np.sin(x)

    def hardlim(self, x):
        return (np.sign(x) + 1) / 2

    def tribas(self, x):
        return np.maximum(1 - np.abs(x), 0)

    def radbas(self, x):
        return np.exp(-(x**2))

    def sign(self, x):
        return np.sign(x)

    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x

    def linear(self, x):
        return x


class ActivationGradient:
    def leaky_relu_gradient(self, x, h):
        x[h < 0] *= 0.1
        return x

    def relu_gradient(self, x, h):
        x[h < 0] = 0
        return x

    def tanh_gradient(self, x, h):
        return 1 - np.tanh(x)**2

    def linear_gradient(self, x, h):
        return np.ones_like(x)




if __name__ == '__main__':
    import sklearn.datasets as sk_dataset

    def prepare_data(proportion):
        dataset = sk_dataset.load_digits()
        label = dataset['target']
        data = dataset['data']
        n_class = len(dataset['target_names'])
        np.random.seed(1)
        shuffle_index = np.arange(len(label))
        np.random.shuffle(shuffle_index)
        np.random.seed(None)
        train_number = int(proportion * len(label))
        train_index = shuffle_index[:train_number]
        val_index = shuffle_index[train_number:]
        data_train = data[train_index]
        label_train = label[train_index]
        data_val = data[val_index]
        label_val = label[val_index]
        return (data_train, label_train), (data_val, label_val), n_class

    num_nodes = 10  # Number of enhancement nodes.
    regular_para = 0.001  # Regularization parameter.
    weight_random_range = [-1, 1]  # Range of random weights.
    bias_random_range = [0, 1]  # Range of random weights.

    train, val, num_class = prepare_data(0.7)
    rvfl = RVRR(num_nodes, regular_para, num_class, weight_random_range, bias_random_range, activation='relu',
                    same_feature=False,
                    weights_initialization='uniform', links=True, use_bias=True, hidden_layer_norm=False)

    rvfl.train(train[0], train[1], epochs=0, lr=[1,1], cp=0)
    prediction = rvfl.predict(train[0], output_prob=True)
    accuracy = rvfl.eval(val[0], val[1])

    rvfl = RVRR(num_nodes, regular_para, num_class, weight_random_range, bias_random_range, activation='relu',
                    same_feature=False,
                    weights_initialization='uniform', links=True, use_bias=True, hidden_layer_norm=False)
    rvfl.train(train[0], train[1], epochs=5, lr=[1, 1], cp=1)
    prediction = rvfl.predict(train[0], output_prob=True)

    bp_accuracy = rvfl.eval(val[0], val[1])
    print('BP', bp_accuracy)

    print('No BP', accuracy)
