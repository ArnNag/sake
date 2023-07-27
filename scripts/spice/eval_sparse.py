import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from functools import partial
import tqdm
import os
from utils import load_data, NUM_ELEMENTS, SPICEBatchLoader, SparseSAKEEnergyModel, get_f_loss, get_y_loss


def run(path, max_nodes=7200, max_edges=120000, max_graphs=400, train_subset=-1, val_subset=3):
    prefix = path[path.find("sparse")+7:path.rfind("eloss")]
    i_vl, x_vl, edges_vl, f_vl, y_vl, num_nodes_vl, num_edges_vl = load_data(prefix + "spice_val.npz", val_subset)
    print("loaded")

    model = SparseSAKEEnergyModel(num_segments=max_graphs)

    loader = SPICEBatchLoader(i_vl, x_vl, edges_vl, f_vl, y_vl, num_nodes_vl, num_edges_vl, 1, max_edges, max_nodes, max_graphs, NUM_ELEMENTS)

    def predict(params, i_full, x_full):
        total_f_loss = 0
        total_y_loss = 0
        for idx in range(len(loader)):
            i, x, edges, f, y, graph_segments = loader.get_batch(idx)  
            total_f_loss += get_f_loss(model, params, i, x, edges, f, graph_segments)
            total_y_loss += get_y_loss(model, params, i, x, edges, y, graph_segments)
        return total_f_loss, total_y_loss

    from flax.training.checkpoints import restore_checkpoint
    save_path = f"val{path}"
    os.mkdir(save_path)
    print("save_path: ", save_path)
    with open(os.path.join(save_path, "losses"), "x") as losses:
        losses.write("Validation force loss\tValidation energy loss\n")
        for checkpoint in sorted(os.listdir(path)):
            losses.write(checkpoint + ": ")
            checkpoint_path = os.path.join(path, checkpoint, "checkpoint")
            print("checkpoint_path: ", checkpoint_path)
            state = restore_checkpoint(checkpoint_path, None)
            params = state['params']
            total_f_loss, total_y_loss = predict(params, i_vl, x_vl) 
            # jnp.save(os.path.join(save_path, f"{checkpoint}_energies"), y_vl_hat)
            # jnp.save(os.path.join(save_path, f"{checkpoint}_forces"), f_vl_hat)
            losses.write(f"{total_f_loss}\t{total_y_loss}\n")


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
