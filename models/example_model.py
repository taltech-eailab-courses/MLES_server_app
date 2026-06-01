from pydantic import BaseModel


model_a_title="Model A test"
model_a_description="This is a test model for demonstration purposes."

model_b_title="Model B test"
model_b_description="This is another test model for demonstration purposes."

class SamplePredictIn(BaseModel):
    x: float

class SamplePredictOut(BaseModel):
    data: str


class ModelA:
    def predict(self, x: float) -> float:
        return 2 * x + 1

class ModelB:
    def predict(self, x: float) -> float:
        return 0.5 * x - 3
