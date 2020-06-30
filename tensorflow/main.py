
import tensorflow as tf


# 常量
t1 = tf.constant([[1,2],[3,4]])
t2 = tf.constant([[3,4],[5,6]])


result =  t1+t2
sess = tf.Session()  #图计算
print(result)
print(sess.run(result))

#变量
v1 =  tf.Variable(3,name='v1')
#占位符
p1 = tf.placeholder(shape=None,dtype='float',name='ppp')

print(v1)

#变量初始化
init1 = tf.global_variables_initializer()

#实现一个小例子
a = tf.constant(1)
p = tf.placeholder(shape=None,dtype='int32',name='p1')
c= a + p
with tf.Session() as sess:
    print(sess.run(c,feed_dict={p:5}))


