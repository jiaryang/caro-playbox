import tensorflow as tf
from rpdTracerControl import rpdTracerControl
rpdTracerControl.setFilename(name = f"tf_trace.rpd", append=False)
rpd_profile = rpdTracerControl()

rpd_profile.start()
# Define matrix dimensions
m, n, k = 1024, 32, 512

# Set a random seed for reproducibility
tf.random.set_seed(0)

# Create random matrices A and B with the same shapes
A = tf.random.uniform((m, n))
B = tf.random.uniform((n, k))

C = tf.matmul(A, B)
rpd_profile.stop()

print(C.shape)

