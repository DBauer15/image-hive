from app.Loading.Loading import Loading
from app.Clustering.Clustering import Clustering
from app.GraphLayout.GraphLayout import GraphLayout
import os.path as path
import os

IMAGE_FOLDER = 'images3'
IMAGE_FILE_TYPE = 'jpeg'
IMAGE_WIDTH = 127

CLUSTERS = 6
SAMPLES_PER_CLUSTER = 6

LAYOUT_DELTA = 0
COMPOSITION_DELTA = 0
COMPOSITION_SIZE = 100

def main():

    # Assemble pipeline

    # LOADING
    source_path = path.join(os.getcwd(), IMAGE_FOLDER)
    loading = Loading(None, source_path=source_path, source_file_type=IMAGE_FILE_TYPE, resize_width=IMAGE_WIDTH)

    # CLUSTERING
    clustering = Clustering(loading, num_clusters=CLUSTERS, num_samples_per_cluster=SAMPLES_PER_CLUSTER)

    # GRAPH LAYOUT
    graph_layout = GraphLayout(clustering, layout_delta=LAYOUT_DELTA, composition_delta=COMPOSITION_DELTA, composition_size=COMPOSITION_SIZE)

    # Run pipeline
    print('+++++++++ RUN STARTED +++++++++\n')
    graph_layout.run()
    print('+++++++++ RUN FINISHED +++++++++')

main()
