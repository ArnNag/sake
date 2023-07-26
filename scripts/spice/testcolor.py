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

    def setup(self):
        self.dense = nn.Dense(self.out_features)
        self.mean = self.variable("coloring", "mean", lambda: 0.)
        self.std = self.variable("coloring", "std", lambda: 1.)

    def coloring(self, x):
        return self.std * x + self.mean

    def __call__(self, x):
        x = self.dense(x)
        x = self.coloring(x)
        return x


key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
model = DenseWithBias(10)
variables = model.init(key, x)
coloring = variables['coloring']
params = variables['params']
new_var = variables.copy({'coloring': {'mean': 2.0, 'std': 3.0}})
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
    apply_fn=model.apply, params=new_var, tx=optimizer,
)

print(state.opt_state)

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
