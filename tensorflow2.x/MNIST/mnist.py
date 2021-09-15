import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from PIL import Image
import random
import os


#加载数据
(xx, ys),_ = datasets.mnist.load_data()


xs = tf.convert_to_tensor(xx, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)


network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
#记录网络
network.summary()
#优化器
optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

for step, (x,y) in enumerate(db):
    #记录梯度
    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28*28))
        out = network(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.square(out-y_onehot)
        loss = tf.reduce_sum(loss) / 32
    acc_meter.update_state(tf.argmax(out, axis=1), y)
    grads = tape.gradient(loss, network.trainable_variables)
    #优化参数
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 500==0:
        #打印准确率
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
#测试10张图片
try:
    os.chdir('./tensorflow2.x/MNIST')
except:
    pass

for i in range(0,10):
    img = Image.fromarray(xx[random.randint(0,60000 - 1)])
    img.save(f'./predict {tf.argmax(network(tf.reshape(tf.convert_to_tensor(img, dtype=tf.float32) / 255., (-1, 28*28))),1)}.jpg')











