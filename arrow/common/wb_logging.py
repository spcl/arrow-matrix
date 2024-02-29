import os
import uuid
from pathlib import Path
from typing import Union
from mpi4py import MPI
import time
import random
from arrow.common import utils
import pickle

__HAS_WB: bool = False
__LOGS = []
__LOG_COMM: Union[MPI.Comm, None] = None
__ITERATION_DATA: dict = {}
__CONFIG: dict = {}
try:
    import wandb
except:
    __HAS_WB = False

def try_acquire_lock(lockfile):
    try:
        fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        return True
    except FileExistsError:
        return False

def release_lock(lockfile):
    if lockfile is not None and os.path.exists(lockfile):
        print("RELEASING LOCK", flush=True)
        os.remove(lockfile)

def acquire_lock(lockfile):
    tries = 0
    max_tries = 5
    max_wait = 60
    cur_wait = 1
    while not try_acquire_lock(lockfile) and tries < max_tries:
        print("WAITING FOR LOCK", flush=True)
        waiting_time = random.uniform(0.05, cur_wait)
        tries += 1
        time.sleep(waiting_time)
        cur_wait = min(cur_wait * 2, max_wait)
    success = tries < max_tries
    print(f"ACQUIRED LOCK {success}", flush=True)
    return success

def set_iteration_data(data: dict):
    """
    This data will be added to any subsequent call to log before the next call to set_iteration_data
    :param data:
    :return:
    """
    global __ITERATION_DATA
    __ITERATION_DATA = data.copy()
    if __LOG_COMM is not None:
        utils.mpi_print(__LOG_COMM.Get_rank(), str(data))

def log(data: dict):
    global __ITERATION_DATA
    data.update(__ITERATION_DATA)
    __LOGS.append(data)
    if __LOG_COMM is not None:
        utils.mpi_print(__LOG_COMM.Get_rank(), str(data))


def finish():
    data = __LOGS
    all_data = __LOG_COMM.gather(data, 0)

    if __HAS_WB:

        if __LOG_COMM.Get_rank() == 0:
            for i, log in enumerate(all_data):
                for item in log:
                    item['rank'] = i
                    wandb.log(item)

            wandb.finish()

        # Wait for root...
        __LOG_COMM.Barrier()
    elif __LOG_COMM.Get_rank() == 0:
        # Generate a unique ID for this run
        algorithm = __CONFIG["algorithm"]
        dataset = __CONFIG["dataset"]

        for i, log in enumerate(all_data):
            for item in log:
                item['rank'] = i
        flatten = lambda l: [x for sublist in l for x in sublist]
        all_data = flatten(all_data)

        run_id = f"{algorithm}.{dataset}." + str(uuid.uuid1())
        # Log the data to a file
        log_path_pickle = Path(f"./logs/{run_id}.pickle")
        log_path_txt = Path(f"./logs/{run_id}.txt")
        log_path_config = Path(f"./logs/{run_id}.config")
        log_path_config_pickle = Path(f"./logs/{run_id}.config.pickle")
        log_path_pickle.parent.mkdir(parents=True, exist_ok=True)
        # Pickle the data
        with open(log_path_pickle, "wb") as f:
            pickle.dump(all_data, f)
        # Write the data to a text file
        with open(log_path_txt, "w") as f:
            # Write the data as a string
            f.write(str(all_data))
        # Write the config to a text file
        with open(log_path_config, "w") as f:
            # Write the data as a string
            f.write(str(__CONFIG))
        # Pickle the config
        with open(log_path_config_pickle, "wb") as f:
            pickle.dump(__CONFIG, f)


def init_run(config: dict, online: bool = False):
    lock_path = Path("~/.wandb_lock").expanduser()
    locked = False
    try:
        locked = acquire_lock(lock_path)
        wandb.init(
            # set the wandb project where this run will be logged
            project="spmm-mpi",
            # track hyperparameters and run metadata
            config=config,
            mode="online" if online else "offline",
            reinit=True,
            tags=[config['algorithm'], config['device'], config['dataset']]
        )
    finally:
        if locked:
            release_lock(lock_path)

def log_local_runs(path: Path):
    # Iterate over all files in the directory that match .config.pickle
    for config_path in path.glob("*.config.pickle"):
        base_path = str(config_path.parent / Path(config_path.stem).stem)
        indicator_file = Path(base_path + ".logged")
        if not indicator_file.exists():
            # Load the config
            with open(config_path, "rb") as f:
                config = pickle.load(f)
            # Load the data
            with open(base_path + ".pickle", "rb") as f:
                print(f"Loading {base_path}.pickle", flush=True)
                data = pickle.load(f)
            # Log the data
            if len(data) > 0 and isinstance(data[0], dict):
                init_run(config, online=False)
                print(data)
                for item in data:
                    wandb.log(item)
                wandb.finish()
                # Create an indicator file to mark the current fiel as logged
                indicator_file.touch()
            else:
                print(f"Skipping {base_path}. No data found.")
        else:
            print(f"Skipping {base_path}. Already logged.")


def wandb_init(comm: MPI.Comm,
               dataset,
               n_features,
               iterations,
               device,
               algorithm,
               block_width,
               wandb_api_key: str = None
               ):
    """
    Initializes a wandb logging run.
    If None is returned, something went wrong and you should not log anything.
    :param comm:
    :param dataset:
    :param n_features:
    :param iterations:
    :param device:
    :param algorithm:
    :param block_width:
    :param wandb_api_key:
    :return:
    """
    global __LOG_COMM
    global __CONFIG
    __LOG_COMM = comm
    rank = comm.Get_rank()
    utils.mpi_print(rank, "HAS_WB False")
    utils.mpi_print(rank, "LOGGING TO FILE")
    dataset_name = (dataset.split('/'))[-1] if dataset is not None else "synthetic"
    __CONFIG = {
        "dataset": dataset_name,
        "width": block_width,
        "n_features": n_features,
        "iterations": iterations,
        "device": device,
        "ranks": comm.Get_size(),
        "host": "NA",
        "algorithm": algorithm,
    }
    set_iteration_data({})
    utils.mpi_print(rank, str({"dataset": dataset_name, "width": block_width, "n_features": n_features, "iterations": iterations,
           "device": device, "ranks": comm.Get_size(), "host": "NA", "algorithm": algorithm}))
    return None

