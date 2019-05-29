from kipoi.model import BaseModel

import faker


class DummyModel(BaseModel):

    def __init__(self, foo):
        self.foo = foo
    def predict_on_batch(self, x):
        return x.sum()
