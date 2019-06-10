from app.Module import Module
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class FeatureClusters(Module):
    """Applies k-means clustering to the given input features.

    This class uses the feature extraction data of previous modules to partition the data into a
    predefined number of clusters.

    Attributes:
        num_clusters: Number of clusters to split into (int)
    """
    def __init__(self, prev_module, num_clusters):
        super().__init__('FeatureClusters', prev_module)
        self._num_clusters = num_clusters

    def run(self):
        super().run()
        features = self._data['features']
        features = np.array(features)
        print('Clustering {} images in {} clusters'.format(len(features), self._num_clusters))
        kmeans = KMeans(n_clusters=self._num_clusters, random_state=0).fit(features)

        self._result = {
            'images': self._data['images'],
            'features': self._data['features'],
            'labels': kmeans.labels_,
            'centers': kmeans.cluster_centers_,
            'kmeans': kmeans,
        }

    def visualize(self):
        result = self.get_module_results()
        images = result['images']
        labels = result['labels']
        n_images = len(images)
        n_unique_labels = len(np.unique(labels))

        img_counts = []

        plt.figure()
        for i in range(n_unique_labels):
            img_count = 0
            for j in range(n_images):
                if labels[j] == i:
                    #img = cv2.cvtColor(result['images'][j], cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(result['images'][j], cv2.COLOR_BGR2RGB)
                    extent = [img_count*64, (img_count+1)*64, i*64, (i+1)*64]
                    plt.imshow(img, origin='upper', extent=extent, cmap='gray')
                    img_count += 1
            print('{} images with label {}'.format(img_count, i))
            img_counts.append(img_count)

        xextent = np.max(np.array(img_counts))

        plt.axis([0, xextent*64, 0, n_unique_labels*64])
        plt.savefig('graph.pdf', dpi=1200)
        plt.show()
