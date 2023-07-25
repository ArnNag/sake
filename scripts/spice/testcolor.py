import jax
import flax
from flax import traverse_util
import flax.linen as nn
from flax.training import orbax_utils
from orbax import checkpoint
from flax.training.train_state import TrainState
import optax

class DenseWithBias(nn.Module):
    out_features: int
    offset: int

    def setup(self):
        self.dense = nn.Dense(self.out_features)
        self.offset_var = self.variable("coloring", "offset", lambda: self.offset)
        jax.lax.stop_gradient(self.offset_var.value)

    def __call__(self, x):
        x = self.dense(x)
        return x + self.offset_var.value


key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
model = DenseWithBias(10, 1.)
variables = model.init(key, x)
coloring = variables['coloring']
params = variables['params']
new_var = variables.copy({'coloring': {'offset': 2.0}})
print(new_var['coloring'])
resultone = model.apply(variables, x)
resulttwo = model.apply(new_var, x)
print(resultone)
print(resulttwo)    



partition_optimizers = {'trainable': optax.adam(5e-3), 'frozen': optax.set_to_zero()}
param_partitions = flax.core.freeze(traverse_util.path_aware_map(
  lambda path, v: 'frozen' if 'coloring' in path else 'trainable', variables))
optimizer = optax.multi_transform(partition_optimizers, param_partitions)
state = TrainState.create(
    apply_fn=model.apply, params=variables, tx=optimizer,
)

def loss_fn(variables, x):
    return jax.numpy.sum(model.apply(variables, x))


def step_with_loss(state, x):
    params = state.params
    grads = jax.grad(loss_fn)(params, x)
    state = state.apply_gradients(grads=grads)
    print(state.params['coloring'])
    return state

def epoch(state, x):
    for i in range(10):
        state = step_with_loss(state, x)
    return state

epoch(state, x)

# PURE_CKPT_DIR = './pure_ckpt'

# mngr = checkpoint.CheckpointManager(PURE_CKPT_DIR, {'state': checkpoint.PyTreeCheckpointer(), 'other': checkpoint.PyTreeCheckpointer()})
# save_args = orbax_utils.save_args_from_target(state)
# mngr.save(0, {'state': state, 'other': 1})
# # ckptr.restore(PURE_CKPT_DIR, item=TARGET_PYTREE,
              # restore_args=flax.training.orbax_utils.restore_args_from_target(TARGET_PYTREE, mesh=None))
