from app.Loading.ImageSource import ImageSource
from app.Loading.ImageLoader import ImageLoader
from app.Clustering.ImageFeaturesHistogram import ImageFeaturesHistogram
from app.Clustering.FeatureClusters import FeatureClusters
from app.Clustering.ClusterSizeReduction import ClusterSizeReduction
from app.Clustering.DimReductionPCA import DimReductionPCA
from app.Clustering.DimReductionTSNE import DimReductionTSNE
from app.Clustering.ClusterVoronoiTesselation import ClusterVoronoiTesselation
import os.path as path
import os

CLUSTERS = 5
SAMPLES_PER_CLUSTER = 8

def main():
    source_path = path.join(os.getcwd(), 'images')
    image_source = ImageSource(source_path)
    image_loader = ImageLoader(image_source)
    image_features = ImageFeaturesHistogram(image_loader)
    feature_clusters = FeatureClusters(image_features, num_clusters=CLUSTERS)
    cluster_size_reduction = ClusterSizeReduction(feature_clusters, num_elements_per_cluster=SAMPLES_PER_CLUSTER)
    if SAMPLES_PER_CLUSTER > 10:
        dim_reduction = DimReductionTSNE(cluster_size_reduction)
    else:
        dim_reduction = DimReductionPCA(cluster_size_reduction)
    cluster_voronoi_tesselation = ClusterVoronoiTesselation(dim_reduction)

    cluster_voronoi_tesselation.run()
    dim_reduction.visualize()


main()
