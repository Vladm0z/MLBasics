if __name__ == '__main__':
    import tensorflow.compat.v1 as tf
    tf.enable_eager_execution()
    import tensorflow_datasets as tfds

#import tensorflow.compat.v1 as tf 
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow_datasets


class MyDS(object):
        class SubDS(object):
            import numpy as np
            def __init__(self, ds, *, one_hot):
                np = self.__class__.np
                self.ds = [e for e in ds.as_numpy_iterator()]
                self.sds = {(k + 's') : np.stack([
                    (e[k] if len(e[k].shape) > 0 else e[k][None]).reshape(-1) for e in self.ds
                ], 0) for k in self.ds[0].keys()}
                self.one_hot = one_hot
                if one_hot is not None:
                    self.max_one_hot = np.max(self.sds[one_hot + 's'])
            def _to_one_hot(self, a, maxv):
                np = self.__class__.np
                na = np.zeros((a.shape[0], maxv + 1), dtype = a.dtype)
                for i, e in enumerate(a[:, 0]):
                    na[i, e] = True
                return na
            def _apply_one_hot(self, key, maxv):
                assert maxv >= self.max_one_hot, (maxv, self.max_one_hot)
                self.max_one_hot = maxv
                self.sds[key + 's'] = self._to_one_hot(self.sds[key + 's'], self.max_one_hot)
            def next_batch(self, num = 16):
                np = self.__class__.np
                idx = np.random.choice(len(self.ds), num)
                res = {k : np.stack([
                    (self.ds[i][k] if len(self.ds[i][k].shape) > 0 else self.ds[i][k][None]).reshape(-1) for i in idx
                ], 0) for k in self.ds[0].keys()}
                if self.one_hot is not None:
                    res[self.one_hot] = self._to_one_hot(res[self.one_hot], self.max_one_hot)
                for i, (k, v) in enumerate(list(res.items())):
                    res[i] = v
                return res
            def __getattr__(self, name):
                if name not in self.__dict__['sds']:
                    return self.__dict__[name]
                return self.__dict__['sds'][name]
        def __init__(self, name, *, one_hot = None):
            self.ds = tfds.load(name)
            self.sds = {}
            for k, v in self.ds.items():
                self.sds[k] = self.__class__.SubDS(self.ds[k], one_hot = one_hot)
            if one_hot is not None:
                maxh = max(e.max_one_hot for e in self.sds.values())
                for e in self.sds.values():
                    e._apply_one_hot(one_hot, maxh)
        def __getattr__(self, name):
            if name not in self.__dict__['sds']:
                return self.__dict__[name]
            return self.__dict__['sds'][name]



#mnist_data = tensorflow_datasets.load('mnist', split='train')
mnist_data = MyDS('mnist', one_hot = 'label')

#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
tf.disable_eager_execution()
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist_data.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist_data.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist_data.test.images, y:mnist_data.test.labels}))

train_neural_network(x)

