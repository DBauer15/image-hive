from app.Module import Module
import cv2


class ImageLoader(Module):
    def __init__(self, prev_module):
        super().__init__('ImageLoader', prev_module)

    def run(self):
        super().run()
        self._result = []
        for file_name in self._data:
            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self._result.append(img)
        print('Loaded {} images'.format(len(self._result)))
        self.has_run = True
