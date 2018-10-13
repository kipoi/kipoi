from kipoi.model import BaseModel


class MyModel(BaseModel):
    def __init__(self, dummy_add=0):
        self.dummy_add = dummy_add

    def predict_on_batch(self, x):
        return self.dummy_add
