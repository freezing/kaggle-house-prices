import tensorflow as tf
import numpy as np

# Neural Network using TensorFlow

# Learning rate setup
START_LEARNING_RATE = 0.001
DECAY_STEPS = 500
DECAY_RATE = 0.96


class NNModel(object):
    def __init__(self, number_of_features,
                 start_learning_rate=START_LEARNING_RATE, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE):
        """Initializes TensorFlow Neural Network with the given config."""
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.learning_rate = tf.train.exponential_decay(start_learning_rate, self.global_step, decay_steps, decay_rate)

        self.input_X = tf.placeholder(tf.float32, [None, number_of_features], name="input_X")
        self.input_Y = tf.placeholder(tf.float32, [None, 1])

        self.dense_1 = tf.layers.dense(inputs=self.input_X, units=20, activation=tf.nn.relu, name="dense_1")
        # self.dropout_1 = tf.layers.dropout(inputs=self.dense_1, rate=0.75, name="dropout_1")

        self.dense_2 = tf.layers.dense(inputs=self.dense_1, units=30, activation=tf.nn.relu, name="dense_2")
        # self.dropout_2 = tf.layers.dropout(inputs=self.dense_2, rate=0.75, name="dropout_2")

        self.dense_3 = tf.layers.dense(inputs=self.dense_2, units=10, activation=tf.nn.relu, name="dense_3")
        # self.dropout_3 = tf.layers.dropout(inputs=self.dense_3, rate=0.75, name="dropout_3")

        self.logits = tf.layers.dense(inputs=self.dense_3, units=1, activation=tf.nn.relu, name="logits")

        self.loss = tf.losses.mean_squared_error(self.input_Y, self.logits)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name="gradient_descent_optimizer")

        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # Prepare TensorFlow session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def fit(self, train_x, train_y, epochs=1, verbose=False, verbose_step=1):
        """Fits the model according to the given data set and number of epochs (iterations)."""
        train_feed_dict = {self.input_X: train_x, self.input_Y: train_y}
        fit_losses = []
        for epoch in range(epochs):
            _, fit_loss, global_step = self.session.run([self.train_step, self.loss, self.global_step],
                                                        feed_dict=train_feed_dict)
            fit_losses.append(fit_loss)

            if verbose and epoch % verbose_step == 0:
                print("Loss[global_step={:d}]={:f}".format(global_step, fit_loss))

        return fit_losses

    def predict(self, x):
        """Predicts values for the given features."""
        return self.session.run(self.logits, feed_dict={self.input_X: x})

    def evaluate_loss(self, x, y):
        """Evaluates the loss for the given data set."""
        return self.session.run(self.loss, feed_dict={self.input_X: x, self.input_Y: y})


