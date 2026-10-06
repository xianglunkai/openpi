
import jax.numpy as jnp
x = jnp.ones((128, 2048))
w = jnp.ones((2048, 256))
y = x @ w
print('OK', float(y[0,0]))