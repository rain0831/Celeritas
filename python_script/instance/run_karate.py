from celeritas.utils.preprocessing.dataset.karate import Karate
import utils.executor as e
import utils.report_result as r
from pathlib import Path

def run_karate(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):

    dataset_name = "karate"

    karate_config_path = Path("./instance/configs_yaml/karate/karate.yaml")

    print("==== Dataset {} is not on local, downloading... =====".format(dataset_name))
    dataset = Karate(dataset_dir / Path(dataset_name))
    dataset.download()
    dataset.preprocess(num_partitions=9, sequential_train_nodes=True) # num_partitions要记得修改！


    for i in range(num_runs):
        e.run_config(karate_config_path, results_dir / Path("karate/celeritas_karate"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "celeritas")


    r.print_results_summary([results_dir / Path("karate/celeritas_karate")])
