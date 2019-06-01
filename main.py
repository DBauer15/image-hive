from app.Loading.Loading import Loading
from app.Clustering.Clustering import Clustering
from app.GraphLayout.GraphLayout import GraphLayout
import os.path as path
import os

CLUSTERS = 5
SAMPLES_PER_CLUSTER = 8

def main():

    # Assemble pipeline

    # LOADING
    source_path = path.join(os.getcwd(), 'images')
    loading = Loading(None, source_path)

    # CLUSTERING
    clustering = Clustering(loading, CLUSTERS, SAMPLES_PER_CLUSTER)

    # GRAPH LAYOUT
    graph_layout = GraphLayout(clustering)

    # Run pipeline
    print('+++++++++ RUN STARTED +++++++++\n')
    graph_layout.run()
    print('+++++++++ RUN FINISHED +++++++++')

main()
