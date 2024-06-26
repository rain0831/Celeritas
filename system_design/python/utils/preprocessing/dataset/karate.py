from pathlib import Path
from celeritas.utils.preprocessing.dataset_base_class import NodeClassificationDataset
from celeritas.utils.preprocessing.download_tools import download_url, extract_file
import numpy as np
from celeritas.utils.preprocessing.converter.to_tensor import TorchEdgeListConverter
from celeritas.utils.configs.constants import PathConstants
from celeritas.utils.preprocessing.ogb_mapping import remap_ogbn
from omegaconf import OmegaConf


class Karate(NodeClassificationDataset):

    def __init__(self, output_directory: Path):

        super().__init__(output_directory)

        self.dataset_name = "karate"
        self.input_edge_list_file = self.output_directory / Path("edge.csv")
        self.input_node_feature_file = self.output_directory / Path("node-feat.csv")
        self.input_node_label_file = self.output_directory / Path("node-label.csv")
        self.input_train_nodes_file = self.output_directory / Path("train.csv")
        self.input_valid_nodes_file = self.output_directory / Path("valid.csv")
        self.input_test_nodes_file = self.output_directory / Path("test.csv")

    def download(self, overwrite=False):

        pass

    def preprocess(self, num_partitions=1, remap_ids=True, splits=None, sequential_train_nodes=False):

        num_nodes = 35
        train_nodes = np.arange(0, 10).astype(np.int32)
        valid_nodes = np.arange(10, 20).astype(np.int32)
        test_nodes = np.arange(20, num_nodes).astype(np.int32)

        converter = TorchEdgeListConverter(
            output_dir=self.output_directory,
            train_edges=self.input_edge_list_file,
            num_partitions=num_partitions,
            columns=[0, 1],
            remap_ids=remap_ids,
            sequential_train_nodes=sequential_train_nodes,
            delim=",",
            known_node_ids=[train_nodes, valid_nodes, test_nodes]
        )
        dataset_stats = converter.convert()

        feature_dim = 128
        num_classes = 40
        features = np.random.uniform(-1, 1, size=(num_nodes, feature_dim)).astype(np.float32)
        labels = np.random.randint(0, num_classes, size=num_nodes).astype(np.int32)
        
        # if remap_ids:
        #     node_mapping = np.genfromtxt(self.output_directory / Path(PathConstants.node_mapping_path), delimiter=",")
        #     train_nodes, valid_nodes, test_nodes, features, labels = remap_ogbn(node_mapping, train_nodes, valid_nodes, test_nodes, features, labels)

        with open(self.train_nodes_file, "wb") as f:
            f.write(bytes(train_nodes))
        with open(self.valid_nodes_file, "wb") as f:
            f.write(bytes(valid_nodes))
        with open(self.test_nodes_file, "wb") as f:
            f.write(bytes(test_nodes))
        with open(self.node_features_file, "wb") as f:
            f.write(bytes(features))
        with open(self.node_labels_file, "wb") as f:
            f.write(bytes(labels))

        # update dataset yaml
        dataset_stats.num_train = train_nodes.shape[0]
        dataset_stats.num_valid = valid_nodes.shape[0]
        dataset_stats.num_test = test_nodes.shape[0]
        dataset_stats.feature_dim = features.shape[1]
        dataset_stats.num_classes = 40

        dataset_stats.num_nodes = dataset_stats.num_train + dataset_stats.num_valid + dataset_stats.num_test

        with open(self.output_directory / Path("dataset.yaml"), "w") as f:
            yaml_file = OmegaConf.to_yaml(dataset_stats)
            f.writelines(yaml_file)

        return dataset_stats

