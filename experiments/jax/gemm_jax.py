import jax
import jax.numpy as jnp
import jax.random as random

from rpdTracerControl import rpdTracerControl
rpdTracerControl.setFilename(name = f"rpd_tracer_output_trace.rpd", append=False)
rpd_profile = rpdTracerControl()

rpd_profile.start()

m, n, k = 1024, 32, 512

key = random.PRNGKey(0)

A = random.uniform(key, (m, n))
B = random.uniform(key, (n, k))

C = jnp.dot(A, B)

print(C.shape)

rpd_profile.stop()
