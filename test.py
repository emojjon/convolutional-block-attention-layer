from cbattention import CBAttention
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization
net = Sequential()
net.add(BatchNormalization(input_shape = (32, 32, 3)))
net.add(Conv2D(32, (3, 3), activation = 'elu', padding='same'))
net.add(BatchNormalization())
net.add(CBAttention())
net.add(BatchNormalization())
net.add(Conv2D(1, (3, 3), activation = 'elu', padding='same'))
net.add(BatchNormalization())
net.add(Flatten())
net.add(Dense(10, activation = 'softmax'))
net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Control net
c_net = Sequential()
c_net.add(BatchNormalization(input_shape = (32, 32, 3)))
c_net.add(Conv2D(32, (3, 3), activation = 'elu', padding='same'))
c_net.add(BatchNormalization())
c_net.add(Conv2D(1, (3, 3), activation = 'elu', padding='same'))
c_net.add(BatchNormalization())
c_net.add(Flatten())
c_net.add(Dense(10, activation = 'softmax'))
c_net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(net.summary())
print('\n    --------\n\n')
print(c_net.summary())

import tensorflow_datasets as tfds

ds = tfds.load('cifar10', split='train', as_supervised=True)
vds = tfds.load('cifar10', split='test', as_supervised=True)
def ohe (item,label):
    label = tf.one_hot(label,10)
    return (item,label)

ds=ds.map(ohe).batch(64)
vds=vds.map(ohe).batch(64)

CBA_weights_pre_training=[]
for w in net.layers[3].weights:
    CBA_weights_pre_training.append(w.numpy())

net.fit(x = ds, epochs=10, verbose=1, validation_data = vds)

CBA_weights_post_training=[]
for w in net.layers[3].weights:
    CBA_weights_post_training.append(w.numpy())

c_net.fit(x = ds, epochs=10, verbose=1, validation_data = vds)

ssd = 0
for i in range(len(CBA_weights_pre_training)):
    ssd += tf.reduce_sum((CBA_weights_pre_training[i] - CBA_weights_post_training[i])**2)
print("Measure of CBA layer training {}".format(ssd))

rcnet = Sequential(name='rcnet')
rcnet.add(BatchNormalization(input_shape = (32, 32, 3)))
rcnet.add(Conv2D(32, (3, 3), activation = 'elu', padding='same'))
rcnet.add(BatchNormalization())
rcnet.add(CBAttention(hidden_c_reduction=4))
rcnet.add(BatchNormalization())
rcnet.add(Conv2D(1, (3, 3), activation = 'elu', padding='same'))
rcnet.add(BatchNormalization())
rcnet.add(Flatten())
rcnet.add(Dense(10, activation = 'softmax'))
rcnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


asnet = Sequential(name='asnet')
asnet.add(BatchNormalization(input_shape = (32, 32, 3)))
asnet.add(Conv2D(32, (3, 3), activation = 'elu', padding='same'))
asnet.add(BatchNormalization())
asnet.add(CBAttention(s_expansion=4))
asnet.add(BatchNormalization())
asnet.add(Conv2D(1, (3, 3), activation = 'elu', padding='same'))
asnet.add(BatchNormalization())
asnet.add(Flatten())
asnet.add(Dense(10, activation = 'softmax'))
asnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


rcasnet = Sequential(name='rcasnet')
rcasnet.add(BatchNormalization(input_shape = (32, 32, 3)))
rcasnet.add(Conv2D(32, (3, 3), activation = 'elu', padding='same'))
rcasnet.add(BatchNormalization())
rcasnet.add(CBAttention(hidden_c_reduction=4, s_expansion=4))
rcasnet.add(BatchNormalization())
rcasnet.add(Conv2D(1, (3, 3), activation = 'elu', padding='same'))
rcasnet.add(BatchNormalization())
rcasnet.add(Flatten())
rcasnet.add(Dense(10, activation = 'softmax'))
rcasnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


for n in [rcnet, asnet, rcasnet]:
    n.CBA_weights_pre_training=[]
    for w in n.layers[3].weights:
        n.CBA_weights_pre_training.append(w.numpy())

    n.fit(x = ds, epochs=10, verbose=1, validation_data = vds)

    n.CBA_weights_post_training=[]
    for w in n.layers[3].weights:
        n.CBA_weights_post_training.append(w.numpy())

    ssd = 0
    for i in range(len(n.CBA_weights_pre_training)):
        ssd += tf.reduce_sum((n.CBA_weights_pre_training[i] - n.CBA_weights_post_training[i])**2)
    print("Measure of CBA layer training for {}: {}".format(n.name, ssd))


