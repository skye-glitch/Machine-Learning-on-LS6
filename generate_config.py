import os
import yaml


def parse_slurm_nodes():
    # Get SLURM environment variables
    node_list = os.environ.get("RANKS")
    num_gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "1"))
    num_machines = int(os.environ.get("NNODES"))
    num_tasks = num_machines * num_gpus_per_node
    machine_rank = os.environ.get("LOCAL_RANK")
    main_rank = os.environ.get("MAIN_RANK")


    # Construct Accelerate config dictionary
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "num_machines": num_machines,
        "mixed_precision": "no",
        "dynamo_backend": "no",
    }

    # Write to YAML file
    with open("multi_config.yaml", "w") as f:
        yaml.dump(config, f)

    print("Accelerate config file 'multi_config.yaml' generated successfully.")
    lines = []
    for node in node_list.split():
        lines.append(f"{node} slots={num_gpus_per_node}")

    with open("hostfile", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("deepspeed host file hostfile generated successfully.")
# Run the function
if __name__ == "__main__":
    parse_slurm_nodes()
