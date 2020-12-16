import tensorflow as tf
from tensorflow.keras.layers import GRU

non_stateful_gru = tf.keras.models.Sequential([
    GRU(5,input_shape=(None,1),name="rnn"),
])

stateful_gru = tf.keras.models.Sequential([
    GRU(5,stateful=True,batch_input_shape=(2,None,1),name="statefull_rnn"),
])

sequence_data = tf.constant([
    [[-4.], [-3.], [-2.], [-1.], [0.], [1.], [2.], [3.], [4.]],
    [[-40.], [-30.], [-20.], [-10.], [0.], [10.], [20.], [30.], [40.]]
], dtype=tf.float32)

print(sequence_data.shape)

out1=non_stateful_gru(sequence_data)
out2=stateful_gru(sequence_data)

print(out1.shape,out2.shape)

print(stateful_gru.get_layer("statefull_rnn").states)
print(stateful_gru.get_layer("statefull_rnn").reset_states())

