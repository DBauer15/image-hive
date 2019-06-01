from app.Module import Module
from sklearn.manifold import TSNE
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DimReductionTSNE(Module):
    def __init__(self, prev_module):
        super().__init__('DimReductionTSNE', prev_module)
        self._coordinate_mapping = None

    def run(self):
        super().run()
        clusters = self._prev_model.get_module_results()
        features = clusters['features']
        centers = clusters['centers']
        features = np.concatenate((features, centers))

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(features)
        tsne1 = tsne_results[:len(clusters['features']), 0]
        tsne2 = tsne_results[:len(clusters['features']), 1]
        tsnec1 = tsne_results[len(clusters['features']):, 0]
        tsnec2 = tsne_results[len(clusters['features']):, 1]

        self._coordinate_mapping = {
            'images': clusters['images'],
            'labels': clusters['labels'],
            'cx': tsnec1,
            'cy': tsnec2,
            'x': tsne1,
            'y': tsne2
        }

    def get_module_results(self):
        return self._coordinate_mapping

    def visualize(self, glyph_size=1):
        result = self.get_module_results()
        plt.figure()
        plt.axis([np.min(result['x']), np.max([result['x']]), np.min(result['y']), np.max(result['y'])])
        plt.scatter(result['x'], result['y'], c=result['labels'], s=0.5, cmap='Dark2')
        for i, image in enumerate(result['images']):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            extent = [result['x'][i]-(glyph_size/2), result['x'][i]+(glyph_size/2), result['y'][i]-(glyph_size/2), result['y'][i]+(glyph_size/2)]
            plt.imshow(img, origin='upper', extent=extent, cmap='gray')
        plt.savefig('graph.pdf', dpi=800)
        plt.show()