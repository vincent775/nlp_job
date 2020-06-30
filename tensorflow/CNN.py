import  tensorflow  as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

def weight_variable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def bias_variable(shape):
    init =  tf.constant(0.1,shape=shape)
    return tf.Variable(init)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


x = tf.placeholder(shape=[None,784],dtype='float32')
y = tf.placeholder(shape=[None,10],dtype='float32')
keep_prob = tf.placeholder(shape=None,dtype='float32')

x_image = tf.reshape(x,[-1,28,28,1])

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
p_pool = max_pool(h_conv1)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(p_pool,w_conv2) + b_conv2)
p_pool2 = max_pool(h_conv2)

out = tf.nn.dropout(p_pool2,keep_prob=keep_prob)
w_fc = weight_variable([7*7*64,1024])
b_fc = weight_variable([1024])

out = tf.reshape(out,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(out,w_fc) + b_fc)


w_fc2 = weight_variable([1024,10])
b_fc2 = weight_variable([10])

logits = tf.matmul(h_fc1,w_fc2)+b_fc2

loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

correct_prediction = tf.equal(tf.arg_max(logits,1),tf.arg_max(y,1))
acc = tf.reduce_mean(tf.cast(correct_prediction,'float32'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch =  mnist.train.next_batch(64)
        _,accuracy = sess.run([train_op,acc],feed_dict={x:batch[0],
                                           y:batch[1],
                                           keep_prob:0.8})
        if i % 20 == 0:
            print('step:',i,'acc:',accuracy)
    test_acc = sess.run(acc,feed_dict={x:mnist.test.images,
                                       y:mnist.test.lable,
                                       keep_prob:1})

    print('test acc:',test_acc)