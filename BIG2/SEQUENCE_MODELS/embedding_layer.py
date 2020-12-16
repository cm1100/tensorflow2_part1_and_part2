import tensorflow as tf

embd_layer =tf.keras.layers.Embedding(input_dim=501,output_dim=16)

sequence=tf.constant([[[0],[1],[5],[500]]])
print(sequence.shape)

sequence_of_embeddings = embd_layer(sequence)
print(sequence_of_embeddings.shape)

print(embd_layer.get_weights()[0].shape)

print(embd_layer.get_weights()[0][14].shape) #embedding vector for the 14th index

embd_layer2=tf.keras.layers.Embedding(input_dim=501,output_dim=16,mask_zero=True)

masked_sequence_of_embeddings=embd_layer2(sequence)
print(masked_sequence_of_embeddings._keras_mask)