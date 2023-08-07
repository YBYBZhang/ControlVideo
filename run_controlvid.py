from time import sleep

from itertools import product

import jaynes
import glob
from params_proto import ParamsProto
from inference import Lucid, generate


machines = [
    # dict(ip="visiongpu50", gpu_id=0),
    dict(ip="visiongpu50", gpu_id=1),
    dict(ip="visiongpu50", gpu_id=2),
    dict(ip="visiongpu50", gpu_id=3),
    dict(ip="visiongpu50", gpu_id=4),
    dict(ip="visiongpu50", gpu_id=5),
    dict(ip="visiongpu50", gpu_id=6),
    dict(ip="visiongpu50", gpu_id=7),

    dict(ip="visiongpu54", gpu_id=0),
    dict(ip="visiongpu54", gpu_id=1),
    dict(ip="visiongpu54", gpu_id=2),
    dict(ip="visiongpu54", gpu_id=3),
    dict(ip="visiongpu54", gpu_id=4),
    dict(ip="visiongpu54", gpu_id=5),
    dict(ip="visiongpu54", gpu_id=6),
    dict(ip="visiongpu54", gpu_id=7),
    dict(ip="visiongpu55", gpu_id=0),
    dict(ip="visiongpu55", gpu_id=1),
    dict(ip="visiongpu55", gpu_id=2),
    dict(ip="visiongpu55", gpu_id=3),
]


class RunArgs(ParamsProto):
    environments = ["Go1Terrain"]  # , "Anyma1Terrain"] # , "Go1"]  # , "Anyma1Terrain"]
    algos = ["PPO_Schulman"]
    seeds = [400, 500, 600]
    sweep = True


#
# checkpoints = [
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/400/checkpoints/model_last.pt",
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/500/checkpoints/model_last.pt",
#     "/alanyu/pql/investigation/Go1Terrain_no_delay/PPO_Schulman/600/checkpoints/model_last.pt",
# ]


def entrypoint(i, prompt, video_path, output_path, sample_vid_name, ):
    print(f"Hey guys! We're on host {i} running environment {env}")
    from ml_logger import logger
    from ml_logger.job import RUN

    RUN.entity = "alanyu"
    RUN.project = "lucid_sim_runs"

    print(f" RUN prefix {prefix} ")
    logger.configure(prefix=prefix)
    filtered_trajs = logger.glob("edges_ego_*")

    # TODO: MAKE START INDEX CONSISTENT WITH ALAN
    for traj_num, video in enumerate(filtered_trajs, start=1):
        generate(prompt, video_path, f"sample_{traj_num:02}", sample_vid_name)


if __name__ == "__main__":

    if RunArgs.sweep:
        runs = list(product(RunArgs.environments, RunArgs.algos, RunArgs.seeds))
    else:
        runs = list(zip(RunArgs.environments, RunArgs.algos, RunArgs.seeds))

    input(
        f"Running the following {len(runs)} configurations: {runs} \n Press enter to continue..."
    )

    prefix = "/lucid_sim/datasets/lucid_sim"  # "/alanyu/pql/investigation_raw/"

    for i, (env, algo, seed) in enumerate(runs):
        if i < len(machines):
            host = machines[i]["ip"]
            visible_devices = f'{machines[i]["gpu_id"]}'
            jaynes.config(
                launch=dict(ip=host),
                runner=dict(gpus=f"'\"device={visible_devices}\"'"),
            )

            print(f"Setting up config {i} on machine {host}")
            # thunk = instr(entrypoint)

            jaynes.run(
                entrypoint, i=i, env=env, seed=seed, prefix=prefix + postfix
            )
        sleep(2)

    jaynes.listen(200)
