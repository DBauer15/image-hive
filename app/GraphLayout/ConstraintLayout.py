from app.Module import Module


class ConstraintLayout(Module):
    def __init__(self, prev_module):
        super().__init__('ConstraintLayout', prev_module)

    def run(self):
        super().run()
