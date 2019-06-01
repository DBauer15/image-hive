class Module:
    def __init__(self, name, prev_module):
        self._name = name
        self._prev_model = prev_module
        self._data = None
        self._result = None
        self.has_run = False

    def run(self):
        if self._prev_model is not None and not self._prev_model.has_run:
            self._prev_model.run()
            self._data = self._prev_model.get_module_results()
        print('Running module ', self._name)

    def get_module_results(self):
        return self._result

    def visualize(self):
        pass