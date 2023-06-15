import h5py
import jax
import jax.numpy as jnp
import time
import os
import json

class SPICELoader:

    def __init__(self, file_path, train_ratio, test_ratio, batch_size=128, seed=2666, max_atom_num=96):
        self.data = h5py.File(file_path)
        self.names = list(self.data.keys())
        self.key = jax.random.PRNGKey(seed)
        self.batch_size = batch_size
        self.max_atom_num = max_atom_num
        n_samples = len(self.names)
        split_idxs = jax.random.shuffle(self.key, jnp.arange(n_samples))
        n_test = int(test_ratio * n_samples)
        n_train = int(train_ratio * n_samples)
        self.test_names = [self.names[i] for i in split_idxs[:n_test]]
        self.train_names = [self.names[i] for i in split_idxs[n_test:n_test + n_train]]
        self.val_names = [self.names[i] for i in split_idxs[n_test + n_train:]]
        if not os.path.isfile('conf_idxs.json'):
            self._make_conf_idxs(self.train_names)
        with open('conf_idxs.json', 'r') as f:
            self.conf_idxs = json.load(f)

    def _make_conf_idxs(self, train_names):
        conf_idxs = []
        for name in train_names:
            for i in range(len(self.data[name]["conformations"])):
                conf_idxs.append((name, i))
        with open('conf_idxs.json', 'w') as f:
            json.dump(conf_idxs, f)

    def _shuffle_confs(self, conf_idxs):
        return jax.random.permutation(self.key, len(conf_idxs))

    def _get_batch(self, train_order, batch_num):
        batch_idxs = train_order[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        batch_confs = []
        batch_atom_nums = []
        batch_energies = []
        for idx in batch_idxs:
            name, conf_idx = self.conf_idxs[idx]
            atom_nums = jnp.array(self.data[name]['atomic_numbers'])
            if len(atom_nums) > self.max_atom_num:  # will lead to nonstatic batch sizes
                continue
            padded_conf = jnp.pad(self.data[name]['conformations'][conf_idx], ((0, self.max_atom_num - len(atom_nums)), (0, 0)))
            batch_confs.append(padded_conf)
            padded_atom_nums = jnp.pad(atom_nums, (0, self.max_atom_num - len(atom_nums)))
            batch_atom_nums.append(padded_atom_nums)
            batch_energies.append(self.data[name]['dft_total_energy'][conf_idx])
        return jnp.array(batch_confs), jnp.array(batch_atom_nums), jnp.array(batch_energies)

    def get_epoch(self):
        train_order = self._shuffle_confs(self.conf_idxs)
        for i in range(len(train_order) // self.batch_size):
            yield self._get_batch(i, train_order)


spice_loader = SPICELoader('SPICE-1.1.3.hdf5', train_ratio=0.8, test_ratio=0.1, max_atom_num=60)
train_order = spice_loader._shuffle_confs(spice_loader.conf_idxs)

print("Starting")
start = time.time()
for j in range(100):
    batch_confs, batch_atom_nums, batch_energies = spice_loader._get_batch(train_order, j)
    # print(batch_confs.shape)
    # print(batch_atom_nums.shape)
    # print(batch_energies.shape)

end = time.time()
print(end - start)
