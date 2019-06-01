class Module:
    def __init__(self, name, prev_module):
        self._name = name
        self._prev_model = prev_module
        self.has_run = False

    def run(self):
        if self._prev_model is not None and not self._prev_model.has_run:
            self._prev_model.run()
        print('Running module ', self._name)

    def get_module_results(self):
        pass

    def visualize(self):
        pass