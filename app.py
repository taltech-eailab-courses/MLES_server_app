from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr
import uvicorn
import time
import numpy as np
from models.example_model import SamplePredictIn, SamplePredictOut, ModelA, ModelB, model_a_title, model_a_description, model_b_title, model_b_description
from models.mnist import mnist_predict, mnist_preprocess, MnistPredictIn, MnistPredictOut, mnist_title, mnist_description
from models.text_recognition import ocr_preprocessing, ocr_predict, OCRPredictIn, OCRPredictOut, text_recognition_title, text_recognition_description, ChunkedOCRRequestIn, ChunkedOCRRequestOut
from models.tinyllama import generate_text, LLAMARequest, llm_prompt_title, llm_prompt_description
from services.temp_file_manager import prepare_storage, store_packet, retrieve_full_data, create_storage


class Greet(BaseModel):
    name: str

app = FastAPI(
    title="Simple MLES APP",
    version="0.1.0",
    description="REST API for my model (also see the Gradio UI at /ui)."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_a = ModelA()
model_b = ModelB()

prefix = "api/v1"

@app.get(f"/{prefix}/healthcheck", tags=["health"])
def healthcheck():
    return {"data": "healthy"}

@app.post(f"/{prefix}/greet", tags=["greet"])
def greet(item: Greet):
    return {"data": f"Hello, {item.name}!"}


@app.post(f"/{prefix}/predict_a", response_model=SamplePredictOut, tags=["inference"])
def run_predict_a(body: SamplePredictIn):
    time.sleep(3) # intentional delay to simulate a long-running task
    return SamplePredictOut(data=str(model_a.predict(body.x)))

@app.post(f"/{prefix}/predict_b", response_model=SamplePredictOut, tags=["inference"])
def run_predict_b(body: SamplePredictIn):
    time.sleep(3) # intentional delay to simulate a long-running task
    return SamplePredictOut(data=str(model_b.predict(body.x)))

@app.post(f"/{prefix}/mnist", response_model=MnistPredictOut, tags=["inference"])
def run_mnist_predict(body: MnistPredictIn):
    # Preprocess the input
    input_data, err = mnist_preprocess(body.x)
    if err is not None:
        raise gr.Error(f"Invalid data: {err}", duration=5)
    
    # Get the prediction
    prediction = mnist_predict(input_data)
    return MnistPredictOut(data=str(prediction))

@app.post(f"/{prefix}/ocr", response_model=OCRPredictOut, tags=["inference"])
def run_ocr(body: OCRPredictIn):
    # Preprocess the input
    print(body.x[:100])  # Print the first 100 characters for debugging
    input_data, err = ocr_preprocessing(body.x)
    if err is not None:
        raise gr.Error(f"Invalid data: {err}", duration=5)

    # Get the prediction
    prediction = ocr_predict(input_data)
    return OCRPredictOut(data=prediction)

@app.post(f"/{prefix}/ocr_binary", response_model=OCRPredictOut, tags=["inference"])
async def run_ocr_binary(request: Request):
    try:
        # Read the raw binary image data
        raw_bytes = await request.body()
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        image_str = ",".join(map(str, arr.tolist()))
        # Preprocess image bytes (modify ocr_preprocessing to accept bytes)
        input_data, err = ocr_preprocessing(image_str)
        if err is not None:
            raise HTTPException(status_code=400, detail=f"Invalid image: {err}")

        # Perform OCR inference
        prediction = ocr_predict(input_data)

        return OCRPredictOut(data=prediction)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"/{prefix}/ocr_chunked", response_model=ChunkedOCRRequestOut, tags=["inference"])
def run_test(body: ChunkedOCRRequestIn):

    x = body.x
    request_id = body.request_id
    packet_index = body.packet_index
    total_packets = body.total_packets

    print("-"* 100)
    print(x[:100])
    print(f"request_id: {request_id}, packet_index: {packet_index}, total_packets: {total_packets}")

    if packet_index == 0:
        create_storage(request_id)
    
    store_packet(x, request_id)
    if packet_index < total_packets - 1:
        return ChunkedOCRRequestOut(data="Large request received.")

    # Last packet, retrieve full data
    full_data = retrieve_full_data(request_id, packet_index, total_packets)
    print(f"Full data length: {len(full_data)}")
    
    input_data, err = ocr_preprocessing(full_data)
    if err is not None:
        raise HTTPException(status_code=400, detail=f"Invalid image: {err}")
    
    # Perform OCR inference
    prediction = ocr_predict(input_data)

    return ChunkedOCRRequestOut(data=prediction)

@app.post(f"/{prefix}/ask_llm", response_model=dict, tags=["inference"])
def run_tinyllama_predict(body: LLAMARequest):
    # Generate text using TinyLlama
    response = generate_text(body.prompt, body.max_tokens)
    return response


# --- Gradio UI (/ui): two simple sections ---
def ui_predict_a(x: float):
    return model_a.predict(float(x))

def ui_predict_b(x: float):
    return model_b.predict(float(x))

def ui_mnist_predict(x: str):
    input_data, err = mnist_preprocess(x)
    if err is not None:
        # Show a toast error in the UI
        raise gr.Error(f"Invalid MNIST input: {err}")

    # Get the prediction
    predicted_digit = mnist_predict(input_data)
    return str(predicted_digit), {"data": predicted_digit}

def ui_ocr_predict(x: str):
    input_data, err = ocr_preprocessing(x)
    if err is not None:
        # Show a toast error in the UI
        raise gr.Error(f"Invalid OCR input: {err}")

    # Get the prediction
    predicted_text = ocr_predict(input_data)
    return str(predicted_text), {"data": predicted_text}

def ui_llama_predict(prompt: str, max_tokens: int = 100):
    response = generate_text(prompt, max_tokens)
    return str(response["data"]), {"data": response["data"]}

with gr.Blocks(title="ML for Embedded System Playground") as demo:
    gr.Markdown("# ML for Embedded System Playground")

    with gr.Blocks(title="Simple model"):
        gr.Markdown(f"## Simple model test")
        with gr.Tab("Model A"):
            gr.Markdown(f"### {model_a_title}\n {model_a_description} The model performs y = 2*x + 1.")
            x_a = gr.Number(label="Input variable (int)")
            y_a = gr.Number(label="Target variable (int)", interactive=False)
            gr.Button("Predict with Model A").click(ui_predict_a, inputs=x_a, outputs=y_a)

        with gr.Tab("Model B"):
            gr.Markdown(f"### {model_b_title}\n{model_b_description} The model performs y = 0.5*x - 3.")
            x_b = gr.Number(label="Input variable (int)")
            y_b = gr.Number(label="Target variable (int)", interactive=False)
            gr.Button("Predict with Model B").click(ui_predict_b, inputs=x_b, outputs=y_b)

    gr.Markdown(f"## {mnist_title}\n{mnist_description}")
    mnist_img = gr.Textbox(label="Input image (flattened 28x28, comma separated uint8 values)", lines=4, value="0,0,0,...,0")
    # Two outputs: nice label widget + raw JSON (for debugging)
    mnist_y_label = gr.Label(label="Prediction / Scores (per class)")
    mnist_y_json  = gr.JSON(label="Raw output")

    gr.Button("Run MNIST Inference").click(
        fn=ui_mnist_predict,
        inputs=mnist_img,
        outputs=[mnist_y_label, mnist_y_json],
    )

    gr.Markdown(f"## {text_recognition_title}\n{text_recognition_description}")
    ocr_img = gr.Textbox(label="Input image (flattened 60x180 grayscale, comma separated uint8 values)", lines=4, value="0,0,0,...,0")
    # Two outputs: nice label widget + raw JSON (for debugging)
    ocr_y_label = gr.Label(label="Prediction / Scores (per class)")
    ocr_y_json  = gr.JSON(label="Raw output")
    gr.Button("Run OCR Inference").click(
        fn=ui_ocr_predict,
        inputs=ocr_img,
        outputs=[ocr_y_label, ocr_y_json],
    )

    gr.Markdown(f"## {llm_prompt_title}\n{llm_prompt_description}")
    llm_prompt = gr.Textbox(label="Input prompt", lines=4, value="Once upon a time")
    llm_max_tokens = gr.Number(label="Max tokens", value=100)
    llm_y_label = gr.Label(label="Generated text")
    llm_y_json  = gr.JSON(label="Raw output")
    gr.Button("Run LLM Inference").click(
        fn=ui_llama_predict,
        inputs=[llm_prompt, llm_max_tokens],
        outputs=[llm_y_label, llm_y_json],
    )

# mount Gradio at /
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # When launched directly (not via Docker CMD), run uvicorn with asyncio loop.
    prepare_storage()
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,    # set to False if you don't want the file watcher
        loop="asyncio", # avoid uvloop/native deps while stabilizing on Jetson
    )

