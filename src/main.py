from src.app.Loading.Loading import Loading
from src.app.Clustering.Clustering import Clustering
from src.app.GraphLayout.GraphLayout import GraphLayout
import os.path as path
import os
import sys


def main():

    # Read args (we need 8)
    if len(sys.argv)-1 == 8:
        print('Using passed argument settings.')
        IMAGE_FOLDER = sys.argv[1]
        IMAGE_FILE_TYPE = sys.argv[2]
        IMAGE_WIDTH = int(sys.argv[3])
        CLUSTERS = int(sys.argv[4])
        SAMPLES_PER_CLUSTER = int(sys.argv[5])
        LAYOUT_DELTA = float(sys.argv[6])
        COMPOSITION_DELTA = float(sys.argv[7])
        COMPOSITION_SIZE = int(sys.argv[8])
    else:
        print('Using default settings.')
        IMAGE_FOLDER = 'images3'
        IMAGE_FILE_TYPE = 'jpeg'
        IMAGE_WIDTH = 127
        CLUSTERS = 6
        SAMPLES_PER_CLUSTER = 6
        LAYOUT_DELTA = 0
        COMPOSITION_DELTA = 0
        COMPOSITION_SIZE = 500

    print('IMAGE_FOLDER: {}'.format(IMAGE_FOLDER))
    print('IMAGE_FILE_TYPE: {}'.format(IMAGE_FILE_TYPE))
    print('IMAGE_WIDTH: {}'.format(IMAGE_WIDTH))
    print('CLUSTERS: {}'.format(CLUSTERS))
    print('SAMPLES_PER_CLUSTER: {}'.format(SAMPLES_PER_CLUSTER))
    print('LAYOUT_DELTA: {}'.format(LAYOUT_DELTA))
    print('COMPOSITION_DELTA: {}'.format(COMPOSITION_DELTA))
    print('COMPOSITION_SIZE: {}'.format(COMPOSITION_SIZE))

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
