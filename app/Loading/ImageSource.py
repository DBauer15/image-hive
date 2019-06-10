from app.Module import Module
import glob
import os.path as path

class ImageSource(Module):
    def __init__(self, source_path, extension):
        super().__init__('ImageSource', None)
        self._source_path = source_path
        self._extension = extension

    def run(self):
        super().run()
        print('Looking for images in path {}'.format(self._source_path))
        print(path.join(self._source_path, '*.jpg'))
        self._result = glob.glob(path.join(self._source_path, '*.'+self._extension))
        print('Found {} images'.format(len(self._result)))
        self.has_run = True
