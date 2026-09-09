# Multilayer_Consensus_GraphLearning_Data

This repository provides the data sets used to evaluate a graph-learning-based consensus-reaching model for multilayer social networks. The model incorporates node-importance information into multilayer opinion learning, learns layer-specific graph representations and cross-layer information, and performs adaptive opinion adjustment while preserving group opinion diversity and limiting excessive minority-group opinion shift.

The released data include the fixed multilayer network structures, node-importance information, and the initial evaluation matrices used for training, validation, and testing.

---

## 📁 Directory Overview

```text
Multilayer_Consensus_GraphLearning_Data/
├── Illustrative Example/
│   ├── network_arrays.npz
│   ├── decision_dataset.npz
│   └── dataset_info.json
│
└── Comparison Experiment/
    ├── Community/
    │   ├── network_arrays.npz
    │   ├── decision_dataset.npz
    │   └── dataset_info.json
    │
    ├── ER Random/
    │   ├── network_arrays.npz
    │   ├── decision_dataset.npz
    │   └── dataset_info.json
    │
    └── Scale Free/
        ├── network_arrays.npz
        ├── decision_dataset.npz
        └── dataset_info.json
```

- `Illustrative Example/`:  
  Contains the data used for the illustrative example. It is intended to demonstrate the complete consensus-reaching process of the proposed model on one representative multilayer social network.

- `Comparison Experiment/Community/`:  
  Contains the data generated on the community-structured multilayer network used in the comparison experiments.

- `Comparison Experiment/ER Random/`:  
  Contains the data generated on the Erdős-Rényi random multilayer network used in the comparison experiments.

- `Comparison Experiment/Scale Free/`:  
  Contains the data generated on the scale-free multilayer network used in the comparison experiments.

Each data folder contains three files:

- `network_arrays.npz`: fixed multilayer network structure and node-level importance information.
- `decision_dataset.npz`: initial evaluation matrices and episode-specific preference-group labels.
- `dataset_info.json`: data-generation and network metadata.

---

## 📄 File Format

### 1. Multilayer Network Structure (`network_arrays.npz`)

The file `network_arrays.npz` stores the fixed multilayer network used by all training, validation, and testing episodes in the corresponding data folder.

It contains the following arrays:

- `node_list`:  
  Global node order used consistently by all network and decision arrays.  
  Shape: `(num_nodes,)`

- `layer_names`:  
  Names of the network layers, such as `L1`, `L2`, and `L3`.  
  Shape: `(num_layers,)`

- `adjacency_matrices`:  
  Adjacency matrices of all network layers.  
  Shape: `(num_layers, num_nodes, num_nodes)`

- `node_importance`:  
  Node-importance values calculated from the multilayer network and used by the proposed model.  
  Shape: `(num_nodes,)`

- `importance_feature`:  
  Node-importance feature representation used as an input feature by the graph-learning model.  
  Its exact shape depends on the implementation used in the experiment.

#### Example loading code

```python
import numpy as np
from pathlib import Path

data_dir = Path("Comparison Experiment/Community")

network_data = np.load(
    data_dir / "network_arrays.npz",
    allow_pickle=False,
)

node_list = network_data["node_list"]
layer_names = network_data["layer_names"].astype(str)
adjacency_matrices = network_data["adjacency_matrices"]
node_importance = network_data["node_importance"]
importance_feature = network_data["importance_feature"]

print("Number of nodes:", len(node_list))
print("Network layers:", layer_names)
print("Adjacency shape:", adjacency_matrices.shape)
print("Node-importance shape:", node_importance.shape)
print("Importance-feature shape:", importance_feature.shape)
```

---

### 2. Initial Evaluation Data (`decision_dataset.npz`)

The file `decision_dataset.npz` stores the initial evaluation matrices used in model training, validation, and testing. Each episode has the same multilayer network structure but a separately generated initial opinion state.

It contains the following arrays:

- `node_list`:  
  Global node order consistent with `network_arrays.npz`.  
  Shape: `(num_nodes,)`

- `O0_train`:  
  Initial evaluation matrices for training episodes.  
  Shape: `(num_train, num_nodes, num_alternatives, num_criteria)`

- `O0_val`:  
  Initial evaluation matrices for validation episodes.  
  Shape: `(num_val, num_nodes, num_alternatives, num_criteria)`

- `O0_test`:  
  Initial evaluation matrices for testing episodes.  
  Shape: `(num_test, num_nodes, num_alternatives, num_criteria)`

- `criterion_weights`:  
  Weights of the evaluation criteria used in the reported experiment.  
  Shape: `(num_criteria,)`

- `kmeans_labels_train`:  
  K-means preference-group labels identified from the initial opinions of the training episodes.  
  Shape: `(num_train, num_nodes)`

- `kmeans_labels_val`:  
  K-means preference-group labels identified from the initial opinions of the validation episodes.  
  Shape: `(num_val, num_nodes)`

- `kmeans_labels_test`:  
  K-means preference-group labels identified from the initial opinions of the testing episodes.  
  Shape: `(num_test, num_nodes)`

- `true_labels_train`:  
  Preference-group labels used internally when generating the training initial-opinion data.  
  Shape: `(num_train, num_nodes)`

- `true_labels_val`:  
  Preference-group labels used internally when generating the validation initial-opinion data.  
  Shape: `(num_val, num_nodes)`

- `true_labels_test`:  
  Preference-group labels used internally when generating the testing initial-opinion data.  
  Shape: `(num_test, num_nodes)`

#### Example loading code

```python
import numpy as np
from pathlib import Path

data_dir = Path("Comparison Experiment/Community")

decision_data = np.load(
    data_dir / "decision_dataset.npz",
    allow_pickle=False,
)

node_list = decision_data["node_list"]
O0_train = decision_data["O0_train"]
O0_val = decision_data["O0_val"]
O0_test = decision_data["O0_test"]
criterion_weights = decision_data["criterion_weights"]

kmeans_labels_train = decision_data["kmeans_labels_train"]
kmeans_labels_val = decision_data["kmeans_labels_val"]
kmeans_labels_test = decision_data["kmeans_labels_test"]

print("Training data shape:", O0_train.shape)
print("Validation data shape:", O0_val.shape)
print("Testing data shape:", O0_test.shape)
print("Criterion weights:", criterion_weights)
```

---

### 3. Data Metadata (`dataset_info.json`)

The file `dataset_info.json` records the main data-generation information associated with each released data set, including:

- network preset;
- number of nodes;
- number of network layers;
- number of alternatives and criteria;
- numbers of training, validation, and testing episodes;
- data-generation random seed;
- difficulty setting;
- node-importance calculation mode and source;
- K-means settings;
- criterion weights;
- basic graph statistics.

This file is provided to make the released data easier to inspect and reproduce without embedding experiment-specific parameters into the raw NumPy arrays.

---

## 🔁 Reconstructing the Multilayer Network

The following code reconstructs each layer as a NetworkX `Graph`.

```python
import numpy as np
import networkx as nx
from pathlib import Path


def rebuild_multilayer_network(network_npz_path):
    data = np.load(network_npz_path, allow_pickle=False)

    node_list = data["node_list"].astype(int)
    layer_names = data["layer_names"].astype(str).tolist()
    adjacency_matrices = data["adjacency_matrices"]

    graphs = {}

    for layer_index, layer_name in enumerate(layer_names):
        graph = nx.from_numpy_array(
            (adjacency_matrices[layer_index] > 0).astype(np.int8)
        )

        # Restore the original global node IDs.
        mapping = {
            array_index: int(node_id)
            for array_index, node_id in enumerate(node_list)
        }
        graph = nx.relabel_nodes(graph, mapping)

        graphs[layer_name] = graph

    return graphs


data_dir = Path("Comparison Experiment/Community")

graphs = rebuild_multilayer_network(
    data_dir / "network_arrays.npz"
)

for layer_name, graph in graphs.items():
    print(
        layer_name,
        "nodes =", graph.number_of_nodes(),
        "edges =", graph.number_of_edges(),
    )
```

---

## 👥 Reconstructing Minority Groups

Minority-group membership is not stored as a fixed raw-data field because it depends on the minority-group threshold `q_min` adopted in a specific experiment.

Given the K-means labels of one episode, the minority-group cluster IDs can be reconstructed as follows:

```python
import numpy as np


def identify_minority_clusters(labels, q_min):
    labels = np.asarray(labels)

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    return [
        int(cluster_id)
        for cluster_id, count in zip(unique_labels, counts)
        if count / len(labels) <= q_min
    ]
```

This design keeps the released initial data independent of parameter-sensitivity settings while allowing the same preference-group partition to be reused under different `q_min` values.

---

## 🚀 Using the Data in the Consensus Model

The following code shows a minimal way to load the released data.

```python
import numpy as np
from pathlib import Path

data_dir = Path("Comparison Experiment/Community")

network_data = np.load(
    data_dir / "network_arrays.npz",
    allow_pickle=False,
)

decision_data = np.load(
    data_dir / "decision_dataset.npz",
    allow_pickle=False,
)

dataset = {
    "node_list": decision_data["node_list"],
    "O0_train": decision_data["O0_train"],
    "O0_val": decision_data["O0_val"],
    "O0_test": decision_data["O0_test"],
    "criterion_weights": decision_data["criterion_weights"],
    "kmeans_labels_train": decision_data["kmeans_labels_train"],
    "kmeans_labels_val": decision_data["kmeans_labels_val"],
    "kmeans_labels_test": decision_data["kmeans_labels_test"],
    "node_importance": network_data["node_importance"],
    "importance_feature": network_data["importance_feature"],
}

graphs = rebuild_multilayer_network(
    data_dir / "network_arrays.npz"
)
```

The model can then use the layer-specific adjacency structures, node-importance features, and the corresponding episode-level initial evaluation matrices to reproduce the consensus-reaching experiments.

---

## 📌 Notes

- All adjacency matrices, node-level attributes, group labels, and evaluation matrices use the same global node order stored in `node_list`.
- The evaluation matrices are organized as `(episode, node, alternative, criterion)`.
- The values in `O0_train`, `O0_val`, and `O0_test` are normalized initial evaluation values before consensus adjustment.
- Within each data folder, the multilayer topology and node-importance information are fixed across training, validation, and testing episodes.
- Different episodes use independently generated initial evaluation matrices and corresponding preference-group labels.
- `kmeans_labels_*` are the preference-group labels actually obtained from the initial evaluations and used for minority-group identification.
- `true_labels_*` are retained only as data-generation metadata; they are not a substitute for the K-means group labels used by the consensus model.
- Minority-group cluster IDs are determined from `kmeans_labels_*` according to the selected `q_min` and are therefore not stored as fixed raw data.
- The illustrative example and the three comparison-network data sets are stored separately so that process demonstration and quantitative comparison can be reproduced independently.

---

## 🧠 Model Context

The released data support a multilayer graph-learning consensus-reaching model with the following main components:

1. **Node-importance-aware multilayer representation learning**: node-importance information is incorporated into graph-based neighborhood learning.
2. **Layer-specific graph representation learning**: different network layers are encoded separately to capture layer-dependent interaction patterns.
3. **Cross-layer information integration**: layer-specific information is adaptively integrated for each decision maker.
4. **Learned neighbor-reference opinion adjustment**: the model learns how much neighborhood reference information should be absorbed during each consensus-adjustment step.
5. **Opinion-diversity preservation**: the optimization objective limits excessive loss of group opinion diversity during consensus reaching.
6. **Minority-group protection**: the model constrains excessive minority-group center shift so that consensus improvement does not simply collapse minority opinions toward the majority.

The consensus process terminates when the target consensus threshold is reached or the maximum number of adjustment steps is met. Final alternative scores and rankings are then calculated from the resulting group evaluations.

---

## 📬 Contact & Citation

If you have any questions regarding the data set or encounter any issues using it, please contact us:

- 📫 **Contact**: 230239729@seu.edu.cn

If you use this data set in your research, please cite the corresponding paper:

> **A Multi-layer Network Graph Learning Consensus-reaching Model Based on the Node Importance and Opinion Diversity**  
> Zhengyi An, et al.

### BibTeX citation

```bibtex
@article{An20XXMultilayerGraphConsensus,
  title   = {A Multi-layer Network Graph Learning Consensus-reaching Model Based on the Node Importance and Opinion Diversity},
  author  = {Zhengyi An and others},
  journal = {XXXXXX},
  year    = {20XX}
}
```

The bibliographic information can be updated after the paper is formally published.
