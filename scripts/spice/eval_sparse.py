import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from functools import partial
import tqdm
import os
import time
from utils import load_data, NUM_ELEMENTS, make_batch_loader, SAKEEnergyModel, get_f_loss, get_y_loss


def run(param_path, val_path, max_nodes=7200, max_edges=120000, max_graphs=1000, train_subset=-1, val_subset=3, seed=0):
    graph_list = load_data(val_path, val_subset)[0]
    print("loaded")

    model = SAKEEnergyModel()

    batch_loader = make_batch_loader(graph_list, seed=seed, max_nodes=max_nodes, max_edges=max_edges, max_graphs=max_graphs, num_elements=NUM_ELEMENTS)

    def predict(params):
        total_f_loss = 0
        total_y_loss = 0
        for graph in batch_loader:
            f_loss = get_f_loss(model, params, graph)
            y_loss = get_y_loss(model, params, graph)
            print("f_loss: ", f_loss)
            print("y_loss: ", y_loss)
            total_f_loss += f_loss
            total_y_loss += y_loss
        return total_f_loss, total_y_loss

    from flax.training.checkpoints import restore_checkpoint
    save_path = f"val_debug_graphs_{max_graphs}_nodes_{max_nodes}_edges_{max_edges}_seed_{seed}_{int(time.time())}"
    os.mkdir(save_path)
    print("save_path: ", save_path)
    with open(os.path.join(save_path, "losses"), "x") as losses:
        losses.write("Checkpoint\tValidation force loss\tValidation energy loss\n")
        for checkpoint in sorted(os.listdir(param_path)):
            losses.write(checkpoint + "\t")
            checkpoint_path = os.path.join(param_path, checkpoint, "checkpoint")
            print("checkpoint_path: ", checkpoint_path)
            state = restore_checkpoint(checkpoint_path, None)
            params = state['params']
            print("params: ", params)
            total_f_loss, total_y_loss = predict(params)
            # jnp.save(os.path.join(save_path, f"{checkpoint}_energies"), y_vl_hat)
            # jnp.save(os.path.join(save_path, f"{checkpoint}_forces"), f_vl_hat)
            losses.write(f"{total_f_loss}\t{total_y_loss}\n")

def scan(param_path, val_path):
    for graphs in range(600, 900, 100):
        for nodes in range(6000, 9000, 1000):
            for edges in range(50000, 80000, 10000):
                for seed in [0, 1]:
                    run(param_path=param_path, val_path=val_path, max_graphs=graphs, max_nodes=nodes, max_edges=edges, seed=seed)



if __name__ == "__main__":
    import sys
    run("_sparse_full_96_dist_nums_eloss_0e+00_subset_3", "small_96_dist_nums_spice_train.npz")
