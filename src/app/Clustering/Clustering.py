from src.app.Module import Module
from src.app.Clustering.ImageFeaturesHistogram import ImageFeaturesHistogram
from src.app.Clustering.FeatureClusters import FeatureClusters
from src.app.Clustering.ClusterSizeReduction import ClusterSizeReduction
from src.app.Clustering.DimReductionPCA import DimReductionPCA
from src.app.Clustering.ClusterVoronoiTesselation import ClusterVoronoiTesselation


class Clustering(Module):
    """Container class for all clustering modules.

    This class accumulates all clustering modules and executes them with defined settings.

    Attributes:
        num_clusters: Number of clusters to generate (int)
        num_samples_per_cluster: Number of images to use per cluster (int)
    """
    def __init__(self, prev_module, num_clusters, num_samples_per_cluster):
        super().__init__('1_CLUSTERING', prev_module)
        self.num_clusters = num_clusters
        self.num_samples_per_cluster = num_samples_per_cluster

    def run(self):
        """Runs all clustering sub-modules.
        """
        image_features = ImageFeaturesHistogram(self._prev_model)
        feature_clusters = FeatureClusters(image_features, num_clusters=self.num_clusters)
        cluster_size_reduction = ClusterSizeReduction(feature_clusters, num_elements_per_cluster=self.num_samples_per_cluster)
        dim_reduction = DimReductionPCA(cluster_size_reduction)
        cluster_voronoi_tesselation = ClusterVoronoiTesselation(dim_reduction)

        cluster_voronoi_tesselation.run()
        self._result = cluster_voronoi_tesselation.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
