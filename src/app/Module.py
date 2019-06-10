class Module:
    """Base class for all modules.

    This class provides basic functionality shared among all modules.
    It automatically runs previous modules when needed and provides access to module in- and output data.

    Attributes:
        _name: Name of this module (string)
        _prev_model: Previous module in the chain of modules (Module)
        _data: Input data for this modules. Comes from previous modules (object)
        _result: Result of this module after running (object)
        has_run: Boolean indicating if this module has already been processed (bool)
    """
    def __init__(self, name, prev_module):
        self._name = name
        self._prev_model = prev_module
        self._data = None
        self._result = None
        self.has_run = False

    def run(self):
        """Runs the module.

        This method runs the previous module if needed and stores its result to make it available to inheriting classes.
        """
        if self._prev_model is not None and not self._prev_model.has_run:
            self._prev_model.run()
            self._data = self._prev_model.get_module_results()
        print('Running module ', self._name)

    def get_module_results(self):
        """Returns this module's results.

        Returns:
            Results from this model's run (object)
        """
        return self._result

    def visualize(self):
        """Visualize this module's results.

        Child classes can override this to present some kind of visual output of their run.
        """
        pass