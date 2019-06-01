from app.Module import Module
import numpy as np

class ClusterSizeReduction(Module):
    def __init__(self, prev_module, num_elements_per_cluster):
        super().__init__('ClusterSizeReduction', prev_module)
        self.num_elements_per_cluster = num_elements_per_cluster

    def run(self):
        super().run()
        self._data = self._prev_model.get_module_results()
        num_unique_labels = len(np.unique(self._data['labels']))

        self._result = {
            'images': [],
            'features': [],
            'labels': [],
            'centers': self._data['centers'],
            'kmeans': self._data['kmeans'],
        }

        # Map array indices to corresponding label/euclidian distance
        distances = np.empty((len(self._data['features']), 3))
        for i, feature in enumerate(self._data['features']):
            label = self._data['labels'][i]
            center = self._data['centers'][label]
            feature = np.array(feature)
            distance = np.linalg.norm(center-feature)
            distances[i, 0] = label
            distances[i, 1] = i
            distances[i, 2] = distance

        # Convert to np array and sort by euclidian distance to each centroid
        distances = distances[distances[:, 2].argsort()]

        # Choose num_elements_per_cluster images/labels/features based on min distance
        for i in range(num_unique_labels):
            nearest = distances[distances[:, 0] == i]
            if len(nearest) >= self.num_elements_per_cluster:
                nearest = nearest[:self.num_elements_per_cluster, :]
            for n in nearest:
                index = int(n[1])
                self._result['images'].append(self._data['images'][index])
                self._result['features'].append(self._data['features'][index])
                self._result['labels'].append(self._data['labels'][index])
