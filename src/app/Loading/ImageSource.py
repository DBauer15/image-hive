from src.app.Module import Module
import glob
import os.path as path


class ImageSource(Module):
    """Looks for image files in a given directory.

    This class finds image paths in a given directory with a specific file type.

    Attributes:
        _source_path: The path to the directory in which to look for images(string)
        _extension: File extension to look for (string)
    """
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
