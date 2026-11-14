# MLES App - Machine Learning for Embedded Systems

A FastAPI-based REST API service with integrated Gradio UI for running various machine learning inference models. This application is designed to be deployed on embedded systems and provides endpoints for multiple ML tasks including digit recognition (MNIST), text recognition (OCR), and large language model (LLM) text generation.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
  - [Health & Testing Endpoints](#health--testing-endpoints)
  - [Simple Model Endpoints](#simple-model-endpoints)
  - [ML Inference Endpoints](#ml-inference-endpoints)
- [Gradio UI](#gradio-ui)
- [Architecture](#architecture)
- [Docker Support](#docker-support)
- [License](#license)

## Overview

MLES App is a comprehensive machine learning inference service that provides:
- RESTful API endpoints for various ML models
- Interactive Gradio web interface for testing models
- Support for MNIST digit recognition
- OCR (Optical Character Recognition) capabilities
- Robot task planning with a custom fine-tuned LLM (LoRA, GGUF)
- Chunked data handling for large requests
- Docker deployment support for both standard and Jetson platforms

## Features

- **FastAPI Backend**: High-performance async API framework
- **Gradio UI**: User-friendly web interface at the root path (`/`)
- **Multiple ML Models**:
  - Simple linear models (Model A & B)
  - MNIST digit classification
  - Text recognition (OCR)
  - Robot planner: custom LLM fine-tuned with LoRA
- **CORS Enabled**: Supports cross-origin requests
- **Chunked Upload**: Support for large data transfers via packet-based uploads
- **Health Monitoring**: Built-in health check endpoint
- **Docker Ready**: Includes Docker Compose configurations for easy deployment

## Installation

### Prerequisites

- Docker

## Running the Application
### Development using Docker

#### Standard Docker Compose for x86 or arm computer

```bash
docker compose up mles-app
```

#### For Jetson device

```bash
docker-compose up mles-app-jetson
```

### To verify the application running
```bash
chmod +x test_request.sh

./test_request.sh
```

## API Documentation

The API is versioned and all endpoints are prefixed with `/api/v1`. Interactive API documentation is available at `/docs` (Swagger UI) when the application is running.

### Base URL

```
http://localhost:5000/api/v1
```

---

### Health & Testing Endpoints

#### GET `/api/v1/healthcheck`

**Description**: Check if the service is running and healthy.

**Tags**: `health`

**Response**:
```json
{
  "data": "healthy"
}
```

**Example**:
```bash
curl http://localhost:5000/api/v1/healthcheck
```

---

#### POST `/api/v1/greet`

**Description**: Simple greeting endpoint for testing purposes.

**Tags**: `greet`

**Request Body**:
```json
{
  "name": "string"
}
```

**Response**:
```json
{
  "data": "Hello, {name}!"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/greet \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

### Simple Model Endpoints

These endpoints demonstrate simple linear regression models with intentional 3-second delays to simulate long-running tasks.

#### POST `/api/v1/predict_a`

**Description**: Run inference using Model A (y = 2*x + 1).

**Tags**: `inference`

**Request Body**:
```json
{
  "x": 5.0
}
```

**Response**:
```json
{
  "data": "11.0"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/predict_a \
  -H "Content-Type: application/json" \
  -d '{"x": 5.0}'
```

**Note**: This endpoint includes a 3-second delay to simulate processing time.

---

#### POST `/api/v1/predict_b`

**Description**: Run inference using Model B (y = 0.5*x - 3).

**Tags**: `inference`

**Request Body**:
```json
{
  "x": 10.0
}
```

**Response**:
```json
{
  "data": "2.0"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/predict_b \
  -H "Content-Type: application/json" \
  -d '{"x": 10.0}'
```

**Note**: This endpoint includes a 3-second delay to simulate processing time.

---

### ML Inference Endpoints

#### POST `/api/v1/mnist`

**Description**: Perform digit recognition on MNIST-formatted images (28x28 grayscale).

**Tags**: `inference`

**Model**: MNIST digit classifier using ONNX runtime

**Request Body**:
```json
{
  "x": "0,0,0,0,...,255"
}
```

The `x` field should contain 784 comma-separated uint8 values (28x28 = 784 pixels), representing a flattened grayscale image.

**Response**:
```json
{
  "data": "7"
}
```

The response contains the predicted digit (0-9).

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/mnist \
  -H "Content-Type: application/json" \
  -d '{"x": "0,0,0,0,...,255"}'
```

**Error Handling**:
- Returns error if input data is invalid
- Input must be exactly 784 comma-separated values

---

#### POST `/api/v1/ocr`

**Description**: Perform optical character recognition (OCR) on text images.

**Tags**: `inference`

**Model**: Text recognition model for grayscale images

**Request Body**:
```json
{
  "x": "0,0,0,0,...,255"
}
```

The `x` field should contain comma-separated uint8 values representing a flattened grayscale image (typically 60x180 = 10,800 pixels).

**Response**:
```json
{
  "data": "Recognized text content"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{"x": "0,0,0,0,...,255"}'
```

---

#### POST `/api/v1/ocr_binary`

**Description**: Alternative OCR endpoint that accepts raw binary image data directly (without JSON encoding).

**Tags**: `inference`

**Content-Type**: `application/octet-stream` or `image/*`

**Request Body**: Raw binary image data (uint8 array)

**Response**:
```json
{
  "data": "Recognized text content"
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/ocr_binary \
  -H "Content-Type: application/octet-stream" \
  --data-binary @image.bin
```

**Use Case**: This endpoint is optimized for embedded systems that can send raw image buffers directly without JSON encoding overhead.

---

#### POST `/api/v1/ocr_chunked`

**Description**: Handle large data transfers via chunked packet uploads. This endpoint supports splitting large images or data into multiple packets for reliable transmission from embedded systems with limited memory.

**Tags**: `inference`

**Request Body**:
```json
{
  "x": "0,0,0,...,255",
  "request_id": "unique-request-id-123",
  "packet_index": 1,
  "total_packets": 10
}
```

**Parameters**:
- `x` (string, required): Chunk of comma-separated uint8 values
- `request_id` (string, required): Unique identifier for the complete request
- `packet_index` (integer, required): Current packet number (1-indexed)
- `total_packets` (integer, required): Total number of packets in the request

**Response** (for intermediate packets):
```json
{
  "data": "Large request received."
}
```

**Response** (for final packet):
```json
{
  "data": "Recognized text from complete image"
}
```

**Workflow**:
1. Client sends first packet with `packet_index=1` (creates storage)
2. Client sends subsequent packets with increasing `packet_index`
3. Server stores each packet using the `request_id`
4. On final packet (`packet_index == total_packets`), server:
   - Reassembles all packets
   - Performs OCR inference on complete image
   - Returns recognition result

**Example**:
```bash
# Send packet 1 of 3
curl -X POST http://localhost:5000/api/v1/ocr_chunked \
  -H "Content-Type: application/json" \
  -d '{
    "x": "0,1,2,3,...",
    "request_id": "req-001",
    "packet_index": 1,
    "total_packets": 3
  }'

# Send packet 2 of 3
curl -X POST http://localhost:5000/api/v1/ocr_chunked \
  -H "Content-Type: application/json" \
  -d '{
    "x": "100,101,102,...",
    "request_id": "req-001",
    "packet_index": 2,
    "total_packets": 3
  }'

# Send final packet 3 of 3 (triggers inference)
curl -X POST http://localhost:5000/api/v1/ocr_chunked \
  -H "Content-Type: application/json" \
  -d '{
    "x": "200,201,202,...",
    "request_id": "req-001",
    "packet_index": 3,
    "total_packets": 3
  }'
```

**Use Case**: Ideal for embedded systems with limited RAM that cannot hold entire images in memory, or for unreliable networks where chunked transmission provides better resilience.


---

#### POST `/api/v1/robot_plan`

**Description**: Turn a natural-language request into a JSON plan for a mobile robot, using a
small LLM fine-tuned with LoRA (see the LoRA robot-planner demo).

**Tags**: `inference`

**Model**: Qwen2.5-0.5B-Instruct + LoRA, merged and quantized to GGUF Q4_K_M.
Default file `models/robot-q4_k_m.gguf`; override with the environment variable
`ROBOT_PLANNER_GGUF`. The model is loaded on the first request.

**Request Body**:
```json
{
  "request": "bring the keys to the front door",
  "max_tokens": 200
}
```

**Parameters**:
- `request` (string, required, 1–500 characters): command for the robot
- `max_tokens` (integer, optional, 16–512, default 200): upper limit on generated tokens

**Response** (200):
```json
{
  "data": {"steps": [
    {"action": "navigate", "target": "keys"},
    {"action": "grab", "object": "keys"},
    {"action": "navigate", "target": "front door"},
    {"action": "drop", "object": "keys"}
  ]},
  "valid": true,
  "error": null,
  "raw": "{\"steps\": [...]}",
  "latency_ms": 85.3,
  "model": "robot-q4_k_m.gguf"
}
```

- `valid` is `false` and `data` is `null` when the output is not valid JSON or uses an action other
  than `navigate(target)`, `grab(object)`, `drop(object)` or `stop`; `error` says why.
- `raw` always holds the exact generated text. The latency value above is only an example.
- **503** if the model file is missing.

**Example**:
```bash
curl -X POST http://localhost:5000/api/v1/robot_plan \
  -H "Content-Type: application/json" \
  -d '{"request": "bring the keys to the front door"}'
```

**Environment variables**:
- `ROBOT_PLANNER_GGUF`: path to the GGUF file (default `models/robot-q4_k_m.gguf`)
- `ROBOT_PLANNER_GPU_LAYERS`: layers on GPU/Metal (default `-1` = all)
- `ROBOT_PLANNER_THREADS`: CPU threads (default `6`)

#### Air-writing (IMU trajectories → word image → OCR)

`POST /api/v1/air_writing` — one request per air-written character, same shape as `/ocr_chunked`:

```json
{"request_id": "0a1b2c3d", "packet_index": 0, "total_packets": 5, "x": "x0,y0,z0,x1,y1,z1,..."}
```

- `request_id`: unique string of one word; used as the directory name `.tmp/buffers/{request_id}/` (only `A-Z a-z 0-9 _ -`, max 64).
- `packet_index`: 0-based index of the character, `total_packets`: number of characters (1..5).
- `x`: the 3D trajectory in metres as flattened `x,y,z` triples (world frame, Z up; the writer looks along +Y).

For every request the trajectory is turned into a 2D character image (PCA plane, orientation from the up/reader
directions, not from PCA) and stored as `.tmp/buffers/{request_id}/{packet_index}.png`. Re-sending a packet overwrites it.
When `packet_index == total_packets - 1` arrives, the stored images are combined (same height, bottoms aligned, gaps, never
connected) into a 60×180 grayscale image (`word.png`), the OCR model (`ocr_predict`) runs and the text is returned:
`{"data": "HELLO"}` (earlier packets get `{"data": "Packet 2/5 received."}`). A repeated last packet returns the cached text.
Errors: `{"error": "<code>", "message": "..."}` with 4xx/5xx (`bad_request_id`, `bad_x`, `bad_point_count`,
`character_too_small`/`character_too_large` 422, `missing_packets` 409 + `"missing": [..]`, `total_packets_mismatch` 409, `ocr_failed` 500).
Directories older than 1 h are removed. Logic: `services/air_writing.py`; optional env: `BUFFER_DIR`, `AW_OCR_INVERT=1`, `AW_OCR_UPSCALE`
(default 2). Tests: `pip install pytest httpx && pytest tests/test_air_writing.py`.

---

## Gradio UI

The application includes an interactive web interface powered by Gradio, accessible at the root path:

```
http://localhost:5000/
```

### UI Features

The Gradio interface provides interactive testing for all models:

1. **Simple Models**
   - Model A: Linear function y = 2*x + 1
   - Model B: Linear function y = 0.5*x - 3
   - Input: Single numeric value
   - Output: Calculated result

2. **MNIST Digit Recognition**
   - Input: Flattened 28x28 image (784 comma-separated values)
   - Output: Predicted digit (0-9) with confidence scores
   - Display: Label widget + JSON output

3. **Text Recognition (OCR)**
   - Input: Flattened 60x180 grayscale image (10,800 comma-separated values)
   - Output: Recognized text
   - Display: Label widget + JSON output

4. **LLM Text Generation**
   - Input: Text prompt + max tokens parameter
   - Output: Generated text continuation
   - Display: Label widget + JSON output

### Accessing the UI

Simply navigate to `http://localhost:5000/` in your web browser after starting the application.

---

## Architecture

### Project Structure

```
mles_app/
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── requirements-jetson.txt         # Jetson-specific dependencies
├── docker-compose.yaml             # Docker Compose configuration for development environment
├── docker-compose.prod.yaml        # Docker Compose configuration for production environment
├── test_request.sh                 # API testing script
├── README.md                       # This documentation
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── example_model.py           # Simple linear models
│   ├── mnist.py                   # MNIST digit recognition
│   ├── text_recognition.py        # OCR implementation
│   ├── robot_planner.py           # Custom LLM: request → JSON robot plan
│   ├── tinyllama.py               # Previous TinyLlama module (no longer used)
│   ├── imu_model.py               # IMU sensor model
│   ├── audio_model.py             # Audio processing model
│   ├── mnist-12.onnx              # MNIST ONNX model (FP32)
│   ├── mnist-12-int8.onnx         # MNIST ONNX model (INT8)
│   └── robot-q4_k_m.gguf          # Fine-tuned planner model (not in git)
└── services/                       # Service utilities
    ├── __init__.py
    └── temp_file_manager.py       # Packet storage for chunked uploads
```

### Key Components

1. **FastAPI Application** (`app.py`)
   - REST API endpoints with OpenAPI documentation
   - CORS middleware for cross-origin requests
   - Request validation using Pydantic models
   - Error handling and logging

2. **Model Modules** (`models/`)
   - Encapsulated model implementations
   - Preprocessing and inference logic
   - Model metadata (titles, descriptions)

3. **Services** (`services/`)
   - Temporary file management for chunked uploads
   - Packet storage and retrieval
   - Request ID-based data assembly

4. **Gradio Integration**
   - Mounted at root path (`/`)
   - Interactive UI components
   - Real-time inference testing

### Technology Stack

- **Framework**: FastAPI 0.116.1
- **UI**: Gradio 5.42.0
- **ML Runtime**: ONNX Runtime, llama-cpp-python
- **Server**: Uvicorn with asyncio
- **Containerization**: Docker, Docker Compose
- **Python**: 3.8+

---

## Docker Support

### Development environment

The application includes a `docker-compose.yaml` for standard deployment:

```bash
docker-compose up mles-app --build
```
For Jetson devices (e.g., Jetson Orin Nano Super), specify service:

```bash
docker-compose up mles-app-jetson --build
```

### Production environment

For a production environment, specify production configuration (requires Ngrok configuration):

```bash
docker-compose -f docker-compose.prod.yaml up mles-app ngrok --build
```

The Jetson configuration includes:
- ARM64 architecture optimizations
- CUDA support for GPU acceleration
- Jetson-specific ML runtime libraries

---

## Testing

Use the provided test script to verify API endpoints:

```bash
./test_request.sh
```

Or test manually:

```bash
# Health check
curl http://localhost:5000/api/v1/healthcheck

# Simple prediction
curl -X POST http://localhost:5000/api/v1/predict_a \
  -H "Content-Type: application/json" \
  -d '{"x": 5.0}'

# MNIST inference (requires valid 784-value image data)
curl -X POST http://localhost:5000/api/v1/mnist \
  -H "Content-Type: application/json" \
  -d '{"x": "0,0,0,..."}'
```

---

## Configuration

### Server Configuration

The application runs on:
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `5000`
- **Reload**: `True` (development mode)
- **Loop**: `asyncio` (compatible with Jetson)

### Model Configuration

Models are loaded at application startup:
- MNIST ONNX models located in `models/`
- Fine-tuned robot planner GGUF (`robot-q4_k_m.gguf`) located in `models/`, or set `ROBOT_PLANNER_GGUF`
- Model weights and configuration embedded in model modules

---

## Performance Considerations

1. **Embedded Systems**
   - Use chunked upload (`/ocr_chunked`) for large images
   - Consider INT8 quantized models for faster inference
   - Monitor memory usage with packet-based processing

2. **Network Optimization**
   - Binary OCR endpoint (`/ocr`) reduces JSON overhead
   - Packet-based uploads improve reliability on unstable connections

3. **Model Inference**
   - Simple models include 3-second delays (intentional, for demo)
   - ONNX models provide optimized inference
   - The robot planner uses 4-bit quantized weights (Q4_K_M) for efficiency

---

## Troubleshooting

### Common Issues

**Port already in use**:
```bash
# Check what's using port 5000
lsof -i :5000

# Kill the process or change port in app.py
```

**Missing model files**:
- Ensure `.onnx` and `.gguf` files are in the `models/` directory
- Check model paths in respective model modules

**Memory issues on Jetson**:
- Use chunked upload for large data
- Monitor with `nvidia-smi` or `tegrastats`
- Reduce `max_tokens` for LLM generation

**CORS errors**:
- CORS is enabled for all origins by default
- Check browser console for specific CORS issues

---

## License

[Specify your license here]

---

## Contributing

[Add contribution guidelines here]

---

## Contact

[Add contact information here]

---

## Acknowledgments

- FastAPI framework
- Gradio for interactive ML interfaces
- ONNX Runtime for efficient model inference
- llama.cpp for efficient LLM inference
- Qwen model developers (base model of the robot planner)
