import os 
os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")  # silence the affinity errors

import onnx
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from onnx import numpy_helper
from pydantic import BaseModel
import warnings

warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'")


mnist_title="MNIST model"
mnist_description="ShuffleNet is a deep convolutional network for image classification. ShuffleNetV2 is an improved architecture that is the state-of-the-art in terms of speed and accuracy tradeoff used for image classification."

model_path = 'models/mnist-12.onnx'
model = onnx.load(model_path)
so = ort.SessionOptions()
so.intra_op_num_threads = 2
so.inter_op_num_threads = 1

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# session = ort.InferenceSession(model.SerializeToString())
session = ort.InferenceSession(
    model.SerializeToString(),
    session_options=so,
    providers=providers
)
# session = make_ort_session(model_path)

input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]


class MnistPredictIn(BaseModel):
    x: str

class MnistPredictOut(BaseModel):
    data: str


def get_image(path):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    return img

# --- Load ONNX test input stored as .pb (TensorProto) ---
def load_onnx_pb_input(pb_path):
    """Load an ONNX TensorProto (.pb) test input and return a numpy array."""
    tensor = onnx.TensorProto()
    with open(pb_path, 'rb') as f:
        tensor.ParseFromString(f.read())
    arr = numpy_helper.to_array(tensor)
    # Ensure float32 for ONNXRuntime and add batch dimension if missing
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.ndim == 3:  # e.g., CxHxW -> 1xC xH xW
        arr = np.expand_dims(arr, axis=0)
    return arr

    
def mnist_preprocess(img):
    """Preprocess the MNIST input data.
        returns: preprocessed image as numpy array with shape (1, 1, 28, 28)
                    values are in range [0.0, 1.0]
    """
    if isinstance(img, str):
        img = np.array([max(min(float(i) / 255.0, 1.0), 0.0) for i in img.split(",")]).astype(np.float32)
    else:
        return None, "Invalid data type received."

    if img.ndim == 1:
        # Assuming the input is a flattened 28x28 image
        if img.size != 28*28:
            return None, "Invalid image size. Data must be comma separated and with a shape 1x784"
        img = img.reshape(1, 1, 28, 28)
    elif img.ndim == 2:
        if img.shape != (28, 28):
            return None, "Invalid image size. Data must be comma separated and with a shape 1x784"
        img = img.reshape(1, 1, 28, 28)
    else:
        return None, "Invalid image format. Data must be comma separated and with a shape 1x784"

    return img, None


def mnist_predict(input_data: np.ndarray):
    inputs = {input_names[0]: input_data}
    outputs = session.run(output_names, inputs)
    outputs = np.squeeze(outputs)
    return np.argmax(outputs)
    

if __name__ == "__main__":
    import os
    from torchvision import datasets, transforms
    from collections import defaultdict

    transform = transforms.Compose([
        transforms.ToTensor(),                 # convert to tensor (0..1)
    ])

    mnist_test = datasets.MNIST(root="./.data/mnist", train=False, download=True, transform=transform)

    digit_data = defaultdict(list)

    for img, label in mnist_test:
        digit_data[label].append(img)

    # Convert lists to torch tensors (shape: [N, 1, 28, 28])
    for digit in digit_data:
        digit_data[digit] = torch.stack(digit_data[digit])

    # Save numpy arrays for testing ONNX inference
    for digit, tensor in digit_data.items():
        np_arr = tensor.numpy()[0]
        input_tensor = torch.tensor(np_arr).unsqueeze(0)  # Add batch dimension
        
        # with open(f".data/mnist/digit_{digit}.txt", "w") as f:
        #     for row in np_arr:
        #         f.write(" ".join(map(str, row)) + "\n")

        # with open(f".data/mnist/flatten_str/{digit}.txt", "w") as f:
        #     f.write(",".join(map(str, np_arr.flatten())) + "\n")

    digit_list = [digit for digit in os.listdir(".data/mnist/flatten_str") if digit.endswith(".txt")]
    for digit_file in digit_list:
        with open(os.path.join(".data/mnist/flatten_str", digit_file), "r") as f:
            d = digit_file.split(".")[0]
            digit_str = f.read()
            input_tensor, err = mnist_preprocess(digit_str)
            if err:
                print(f"Error processing digit {digit_file}:", err)
            else:
                print(f"Processed digit {digit_file}:", input_tensor.shape)

                outputs = mnist_predict(input_tensor)
                print(f"Predictions for digit {d}:", outputs)