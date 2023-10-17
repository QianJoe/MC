import collections
from typing import Union, Tuple, Dict

import torch
from torch import Tensor

from models.anomaly_map import AnomalyMapGenerator
from models.feature_extractor import FeatureExtractor
import torch.nn.functional as F

class MemoryCore(torch.nn.Module):
    def __init__(self,config):
        """PatchCore anomaly detection class."""
        super(MemoryCore, self).__init__()
        self.backbone = config.model.backbone
        self.layers = config.model.layers
        self.input_size = config.dataset.image_size

        self.feature_extractor = FeatureExtractor(backbone=self.backbone, pre_trained=config.model.pre_trained,
                                                  layers=self.layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self.input_size)

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor
        self.feature_bank = collections.defaultdict(list)

    def forward(self, input_tensor: Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.

        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)


        if self.training:
            output = embedding
            embedding_size = embedding.size(1)
            embedding = embedding.permute(0, 2, 3, 1).reshape(embedding.shape[0], -1, embedding_size)
            if not self.feature_bank:
                embedding = torch.mean(embedding, 0)
                for index, feature in enumerate(embedding):
                    self.feature_bank[index].append(feature)
            else:
                for i in range(embedding.size(1)):
                    features = embedding[:, i]  # [batchsize,1024]
                    bank_features = self.feature_bank[i]
                    bank_features = torch.stack(bank_features)
                    # for bank_feature in bank_features:
                    #distances = torch.mm(features, bank_features.t())
                    distances = torch.nn.functional.pairwise_distance(features.unsqueeze(1), bank_features,p=2)
                    #  distances = torch.cdist(features.unsqueeze(1), bank_features, p=2).squeeze(1)
                    # for distance in distances:
                    distances = torch.sum((torch.mul(distances,distances)),dim=1)
                    #max_not_similarity_index = torch.argmin(distances) // distances.shape[1]
                    max_not_similarity_index = torch.argmax(distances)
                    self.feature_bank[i].append(features[max_not_similarity_index])
        else:
            feature_map_shape = embedding.shape[-2:]
            embedding = self.reshape_embedding(embedding)
            patch_scores = self.nearest_neighbors(embedding=embedding)
            anomaly_map,anomaly_score = self.anomaly_map_generator(
                patch_scores=patch_scores, feature_map_shape=feature_map_shape
            )
            output = (anomaly_map,anomaly_score)

        return output

    def train_end(self) -> None:
        feature_list = []
        for i in range(len(self.feature_bank)):
            feature_list += self.feature_bank[i]

        self.memory_bank = torch.vstack(feature_list)

    def generate_embedding(self, features: Dict[str, Tensor]) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    def nearest_neighbors(self, embedding: Tensor) -> Tensor:
        """Nearest Neighbours using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
        """
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)  # euclidean norm
        patch_scores, _ = distances.topk(k=1, largest=False, dim=1)
        return patch_scores
