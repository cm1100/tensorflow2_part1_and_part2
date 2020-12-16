import tensorflow as tf


x = tf.constant([0,1,2,3],dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y=tf.reduce_sum(x**2)
    z=tf.math.sin(y)
    dz_dy = tape.gradient(z,y)

print(dz_dy)


with tf.GradientTape() as tape:
    tape.watch(x)
    y=tf.reduce_sum(x**2)
    z= tf.math.sin(y)
    dz_dy,dz_dx= tape.gradient(z,[y,x])

print(dz_dy)
print(dz_dx)