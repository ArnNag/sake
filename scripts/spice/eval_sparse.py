import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as onp
import sake
from functools import partial
import tqdm
import os
from utils import load_data, NUM_ELEMENTS, SPICEBatchLoader, SparseSAKEEnergyModel, get_e_pred, get_f_pred


def run(path, max_nodes=3600, max_edges=60000, max_graphs=152, train_subset=-1, val_subset=3):
    prefix = path[path.find("sparse")+7:path.rfind("eloss")]
    print("prefix: ", prefix)
    i_vl, x_vl, edges_vl, f_vl, y_vl, num_nodes_vl, num_edges_vl = load_data(prefix + "spice_val.npz", val_subset)
    print("loaded")

    model = SparseSAKEEnergyModel(num_segments=max_graphs)

    loader = SPICEBatchLoader(i_vl, x_vl, edges_vl, f_vl, y_vl, num_nodes_vl, num_edges_vl, 0, max_edges, max_nodes, max_graphs, NUM_ELEMENTS)

    def predict(params, i_full, x_full):
        y_hat_all = []
        f_hat_all = []
        for idx in range(len(loader)):
            i, x, edges, f, y, graph_segments = loader.get_batch(idx)  
            y_hat_all.append(get_e_pred(model, params, i, x, edges, graph_segments))
            f_hat_all.append(get_f_pred(model, params, i, x, edges, graph_segments))
        y_hat = jnp.concatenate(y_hat_all)
        f_hat = jnp.concatenate(f_hat_all)
        return f_hat, y_hat

    from flax.training.checkpoints import restore_checkpoint
    save_path = f"val{path}"
    os.mkdir(save_path)
    print("save_path: ", save_path)
    with open(os.path.join(save_path, "losses"), "x") as losses:
        for checkpoint in sorted(os.listdir(path)):
            losses.write(checkpoint + ": ")
            checkpoint_path = os.path.join(path, checkpoint, "checkpoint")
            print("checkpoint_path: ", checkpoint_path)
            state = restore_checkpoint(checkpoint_path, None)
            params = state['params']
            f_vl_hat, y_vl_hat = predict(params, i_vl, x_vl) 
            jnp.save(os.path.join(save_path, f"{checkpoint}_energies"), y_vl_hat)
            jnp.save(os.path.join(save_path, f"{checkpoint}_forces"), f_vl_hat)
            losses.write(f"validation energy loss: {sake.utils.bootstrap_mae(y_vl_hat, y_vl)} ")
            losses.write(f"validation force loss: {sake.utils.bootstrap_mae(f_vl_hat, f_vl)} \n")


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
