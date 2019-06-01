from app.Module import Module

class LinkNodes(Module):
    def __init__(self, prev_module):
        super().__init__('LinkNodes', prev_module)
        self._result = None

    def run(self):
        super().run()

    def get_module_results(self):
        return self._result
