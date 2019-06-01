from app.Module import Module
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class FeatureClusters(Module):
    def __init__(self, prev_module, num_clusters):
        super().__init__('FeatureClusters', prev_module)
        self._clusters = None
        self._num_clusters = num_clusters

    def run(self):
        super().run()
        f = self._prev_model.get_module_results()
        features = f['features']
        features = np.array(features)
        print('Clustering {} images in {} clusters'.format(len(features), self._num_clusters))
        kmeans = KMeans(n_clusters=self._num_clusters, random_state=0).fit(features)

        self._clusters = {
            'images': f['images'],
            'features': f['features'],
            'labels': kmeans.labels_,
            'centers': kmeans.cluster_centers_,
            'kmeans': kmeans,
        }

    def get_module_results(self):
        return self._clusters

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
