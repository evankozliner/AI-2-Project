import sys
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import functools
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

def createDataset(data, lookback=1):
    dataX, dataY = [], []
    for i in range(len(data) - lookback - 1):
        a = data[i:i+lookback]
        dataX.append(a)
        dataY.append(data[i+lookback][3])
    return np.array(dataX), np.array(dataY).reshape(len(dataY), 1)

def nextBatch(index, batchSize, data):
    begin = index * batchSize
    end = min((index + 1) * batchSize, len(data))
    return data[(index * batchSize): end]

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class VariableSequenceClassification:

    def __init__(self, data, target, num_hidden=1024, num_layers=2):
        self.data = data        
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.accuracy
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        cell = BasicLSTMCell(self._num_hidden)
        #cell = DropoutWrapper(cell, output_keep_prob = 0.8)
        cell = MultiRNNCell([cell] * self._num_layers)

        output, _ = tf.nn.dynamic_rnn(
            cell,
            self.data,
            dtype=tf.float32,
            #sequence_length=self.length,
        )

        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(self._num_hidden, 1)
        prediction = (tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        #cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.prediction, labels = self.target))
        pred = self.prediction[:,0]
        cross_entropy = tf.reduce_mean(tf.square(tf.subtract(self.target[:,0], pred)))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(self.cost)

    @lazy_property
    def accuracy(self):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target, self.prediction))))
        #correct = tf.equal(
        #    tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        #return tf.reduce_mean(tf.cast(correct, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

def main():
    batch_size = 128;
    #mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
    scalerX = MinMaxScaler(feature_range = (0,1))
    scalerY = MinMaxScaler(feature_range = (0,1))

    financialData = np.load('nflx-data.npy')
    financialData = financialData[:,0:7]
    nextOpen = financialData[1:, 0].reshape(-1,1)
    financialData = financialData[1:, :]
    financialData = np.append(financialData, nextOpen, axis = 1)

    Y = financialData[:,3]

    financialData = scalerX.fit_transform(financialData)
    Y = scalerY.fit_transform(Y.reshape(-1,1))
    
    lookBack = 20
    trainX, trainY = createDataset(financialData, lookBack)

    trainSize = 1000 #int(len(trainY) * 0.50  )
    testSize = len(trainY) - trainSize
    testX, testY = trainX[trainSize: len(trainY), :], trainY[trainSize: len(trainY)]
    trainX, trainY = trainX[0:trainSize,:], trainY[0:trainSize]
    
    #trainX, testX = scalerX.fit_transform(trainX), scalerX.fit_transform(testX)
    #trainY, testY = scalerY.fit_transform(trainY), scalerY.fit_transform(testY)

    data = tf.placeholder('float', [None, lookBack, financialData.shape[1]])
    target = tf.placeholder('float')

    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    bestError = 100
    for epoch in range(100):
        #print(epoch,"\n")
        epoch_loss = 0
        indices = np.random.permutation(len(trainX))
        #for _ in range(int(mnist.train.num_examples/batch_size)):
        for index in range(int(len(trainX)/batch_size)):
            #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            #epoch_x = epoch_x.reshape(batch_size, 28, 28)
            indexValues = nextBatch(index, batch_size, indices)
            epoch_x = trainX[indexValues]
            epoch_y = trainY[indexValues]
            #pred = sess.run(model.prediction, {data:epoch_x, target:epoch_y})
            #pred = scalerY.inverse_transform(pred)
            #cost = 
            sess.run(model.optimize, {data:epoch_x, target:epoch_y})
            cost = sess.run(model.cost, {data:epoch_x, target:epoch_y})
            epoch_loss += cost
            
        #accuracy = sess.run(model.accuracy, {data:mnist.test.images.reshape((-1, 28, 28)), target:mnist.test.labels})
        #prediction = sess.run(model.prediction, {data:mnist.test.images.reshape((-1, 28, 28)), target:mnist.test.labels})
        #result = tf.gather(prediction, 0)
        #print (sess.run(result))
        #print (sess.run(tf.reduce_sum(result)))
        
        
        rmseError = sess.run(model.accuracy, {data:testX, target:testY})          
        print('Epoch {:2d} Error {:3.5f}%'.format(epoch + 1, rmseError))
        #print('Loss:', epoch_loss)
        #if(bestError > rmseError):
        #    bestError = rmseError
        #    print(bestError)
            #saver.save(sess, "/tmp/model.ckpt")
    
    
    #saver.restore(sess, "/tmp/model.ckpt")
    testprediction = sess.run(model.prediction, {data:testX, target:testY})
    #print(testprediction[:,0])   
    result = np.zeros((testprediction.shape[0], 2))
    testprediction = scalerY.inverse_transform(testprediction)
    result[:,1] = testprediction[:,0]
    result[:,0] = scalerY.inverse_transform(testY)[:,0]
    print(result)

    testPlot = np.empty_like(Y)
    testPlot[:,:] = np.nan
    testPlot[len(Y) - 1 - testprediction.shape[0]:len(Y)-1, :] = testprediction

    trainprediction = sess.run(model.prediction, {data:trainX, target:trainY})
    trainprediction = scalerY.inverse_transform(trainprediction)
    trainPlot = np.empty_like(Y)
    trainPlot[:,:] = np.nan
    trainPlot[lookBack:lookBack + len(trainprediction), :] = trainprediction

    plt.plot(scalerY.inverse_transform(Y))
    plt.plot(trainPlot)
    plt.plot(testPlot)
    plt.show()

    print("Hello World")
if __name__ == "__main__":
    sys.exit(int(main() or 0))