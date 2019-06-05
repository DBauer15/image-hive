from app.Module import Module
from app.Clustering.ImageFeaturesHistogram import ImageFeaturesHistogram
from app.Clustering.FeatureClusters import FeatureClusters
from app.Clustering.ClusterSizeReduction import ClusterSizeReduction
from app.Clustering.DimReductionPCA import DimReductionPCA
from app.Clustering.DimReductionTSNE import DimReductionTSNE
from app.Clustering.ClusterVoronoiTesselation import ClusterVoronoiTesselation


class Clustering(Module):
    def __init__(self, prev_module, num_clusters, num_samples_per_cluster):
        super().__init__('1_CLUSTERING', prev_module)
        self.num_clusters = num_clusters
        self.num_samples_per_cluster = num_samples_per_cluster

    def run(self):
        image_features = ImageFeaturesHistogram(self._prev_model)
        feature_clusters = FeatureClusters(image_features, num_clusters=self.num_clusters)
        cluster_size_reduction = ClusterSizeReduction(feature_clusters, num_elements_per_cluster=self.num_samples_per_cluster)
        if self.num_samples_per_cluster > 10:
            dim_reduction = DimReductionTSNE(cluster_size_reduction)
        else:
            dim_reduction = DimReductionPCA(cluster_size_reduction)
        cluster_voronoi_tesselation = ClusterVoronoiTesselation(dim_reduction)

        cluster_voronoi_tesselation.run()
        self._result = cluster_voronoi_tesselation.get_module_results()
        print('+++++++++ ' + self._name + ' DONE +++++++++\n')
