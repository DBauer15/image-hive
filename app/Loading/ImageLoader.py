from app.Module import Module
import cv2


class ImageLoader(Module):
    def __init__(self, prev_module):
        super().__init__('ImageLoader', prev_module)
        self._images = None

    def run(self):
        super().run()
        self._images = []
        file_names = self._prev_model.get_module_results()
        for file_name in file_names:
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self._images.append(img)
        print('Loaded {} images'.format(len(self._images)))
        self.has_run = True

    def get_module_results(self):
        return self._images