import h5py
import jax
import jax.numpy as jnp
import time
import json
import os


class SPICESerializer:

    def __init__(self, data_path, train_ratio, test_ratio, seed=2666, max_atom_num=96):
        self.data = h5py.File(data_path)
        self.names = list(self.data.keys())
        self.key = jax.random.PRNGKey(seed)
        self.max_atom_num = max_atom_num
        self.test_names, self.train_names, self.val_names = self.split(train_ratio, test_ratio)
        self.make_npy(self.test_names, "test")
        self.make_npy(self.val_names, "val")
        self.make_npy(self.train_names, "train")

    def split(self, train_ratio, test_ratio):
        n_samples = len(self.names)
        split_idxs = jax.random.permutation(self.key, n_samples)
        n_test = int(test_ratio * n_samples)
        n_train = int(train_ratio * n_samples)
        test_names = [self.names[i] for i in split_idxs[:n_test]]
        train_names = [self.names[i] for i in split_idxs[n_test:n_test + n_train]]
        val_names = [self.names[i] for i in split_idxs[n_test + n_train:]]
        return test_names, train_names, val_names

    def make_npy(self, names, out_path):
        all_confs = []
        all_atom_nums = []
        all_energies = []
        for name in names:
            atom_nums = jnp.array(self.data[name]['atomic_numbers'], jnp.uint8)
            if atom_nums.shape[0] > self.max_atom_num:
                continue
            conf_arr = self.data[name]['conformations']
            energy_arr = self.data[name]['dft_total_energy']
            all_energies.extend(energy_arr)
            for conf in conf_arr:
                padded_conf = jnp.pad(conf, ((0, self.max_atom_num - conf.shape[0]), (0, 0)))
                padded_atom_nums = jnp.pad(atom_nums, (0, self.max_atom_num - atom_nums.shape[0]))
                all_confs.append(padded_conf)
                all_atom_nums.append(padded_atom_nums)
        jnp.save(out_path + "_confs", jnp.array(all_confs))
        jnp.save(out_path + "_atom_nums", jnp.array(all_atom_nums))
        jnp.save(out_path + "_energies", jnp.array(all_energies))


class SpiceLoader:

    def __init__(self, seed=2666, batch_size=128, mmap_mode=None):
        self.key = jax.random.PRNGKey(seed)
        self.batch_size = batch_size
        self.train_energies = jnp.load("train_energies.npy", mmap_mode=mmap_mode)
        self.train_atom_nums = jnp.load("train_atom_nums.npy", mmap_mode=mmap_mode)
        self.train_confs = jnp.load("train_confs.npy", mmap_mode=mmap_mode)

    def get_epoch(self):
        train_size = len(self.train_energies)
        shuffled_idxs = jax.random.permutation(self.key, train_size)
        for batch_num in range(train_size // self.batch_size):
            batch_start = batch_num * self.batch_size
            batch_end = (batch_num + 1) * self.batch_size
            batch_idxs = shuffled_idxs[batch_start:batch_end]
            batch_atom_nums = self.train_atom_nums[batch_idxs]
            batch_confs = self.train_confs[batch_idxs]
            batch_energies = self.train_energies[batch_idxs]
            yield batch_atom_nums, batch_confs, batch_energies


# spice_serializer = SPICESerializer('SPICE-1.1.3.hdf5', train_ratio=0.8, test_ratio=0.1, max_atom_num=60)
spice_loader = SpiceLoader(mmap_mode="r")
start = time.time()

for batch in spice_loader.get_epoch():
    atom_nums, confs, energies = batch

end = time.time()
print(end - start)