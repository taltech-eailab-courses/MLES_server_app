import os
import numpy as np

BASE_DIR = os.environ.get("BUFFER_DIR", "/app/.tmp/buffers")  # overridable for local runs/tests

def prepare_storage():
    os.makedirs(BASE_DIR, exist_ok=True)

def create_storage(request_id: str):
    tmp_path = os.path.join(BASE_DIR, f"{request_id}.bin")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    with open(tmp_path, "w") as f:
        pass  # Create empty file

def store_packet(data_str: str, request_id: str):
    tmp_path = os.path.join(BASE_DIR, f"{request_id}.bin")
    with open(tmp_path, "a") as f:
        f.write(data_str)

def retrieve_full_data(request_id: str, packet_index: int, total_packets: int) -> str:
    tmp_path = os.path.join(BASE_DIR, f"{request_id}.bin")
    with open(tmp_path, "r") as f:
        text = f.read()

    os.remove(tmp_path)  # Clean up after retrieval
    return text