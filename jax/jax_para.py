import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

devices = mesh_utils.create_device_mesh((2, jax.device_count()//2))
mesh = Mesh(devices, ('data', 'model'))

print(devices)
print(mesh)

def fn(x):
    return x * 2

x = jnp.arange(16).reshape(4, 4)
print(x)

partition_spec = P('data', 'model')

result = shard_map(fn, mesh, partition_spec, partition_spec)(x)
print(result)
