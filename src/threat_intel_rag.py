import hashlib
import requests
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

SALT = os.getenv("HASH_SALT", "default_local_salt")

def hash_sensitive_data(ip_address: str) -> str:
    """
    Data Privacy Layer: 
    Never send raw PII or sensitive network topology (like internal IPs) to an external LLM.
    We use SHA-256 to consistently hash the IP so the analyst can still track repeat 
    offenders without exposing the raw IP to the API.
    """
    return hashlib.sha256((ip_address + SALT).encode()).hexdigest()[:8]

def transform_query_for_rag(anomalous_log: dict, mse_score: float) -> str:
    """
    Query Transformation Layer:
    Converts raw, high-dimensional numeric metrics into a semantic text query
    suitable for vector database retrieval.
    """
    bytes_in = anomalous_log['bytes_in']
    bytes_out = anomalous_log['bytes_out']
    
    # Transform numeric bounds into semantic descriptions
    payload_desc = "massive inbound payload" if bytes_in > 5000 else "standard inbound payload"
    response_desc = "unusually small outbound response" if bytes_out < 1000 else "standard outbound response"
    
    semantic_query = (
        f"Web traffic anomaly with MSE {mse_score:.2f}. "
        f"Traffic exhibits {payload_desc} ({bytes_in} bytes) and {response_desc} ({bytes_out} bytes) "
        f"on port {anomalous_log['dst_port']}. Protocol is {anomalous_log['protocol']}. "
        f"What type of web attack matches this volumetric signature?"
    )
    return semantic_query

def retrieve_historical_context(semantic_query: str) -> str:
    """Simulated RAG Retrieval"""
    if "massive inbound payload" in semantic_query:
        return "Retrieved Context: High inbound bytes coupled with low outbound bytes on HTTPS typically indicates a Buffer Overflow attack or a large SQL Injection payload attempting to bypass WAF rules."
    return "Retrieved Context: Suspicious port scanning or unusual protocol usage."

def generate_security_alert_local(raw_log: dict, mse_score: float):
    # 1. Privacy Masking (This defines safe_src_ip and safe_dst_ip)
    safe_src_ip = hash_sensitive_data(raw_log['src_ip'])
    safe_dst_ip = hash_sensitive_data(raw_log['dst_ip'])
    
    # 2. Query Transformation (This defines search_query)
    search_query = transform_query_for_rag(raw_log, mse_score)
    print(f"--- Transformed Search Query ---\n{search_query}\n")
    
    # 3. Context Retrieval (This defines context)
    context = retrieve_historical_context(search_query)
    
    # 4. Local LLM Generation via Ollama
    prompt = f"""
    You are an expert SOC Analyst. An autoencoder flagged an anomalous web request.
    
    Retrieved Context: {context}
    
    Anomalous Log Details:
    - Source IP (Hashed): {safe_src_ip}
    - Destination IP (Hashed): {safe_dst_ip}
    - Bytes In: {raw_log['bytes_in']}
    - Bytes Out: {raw_log['bytes_out']}
    - MSE Anomaly Score: {mse_score:.2f}
    
    Write a concise, 3-sentence incident response alert.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("--- Final Generated Security Alert (Local LLM) ---")
        print(result['response'])
    except Exception as e:
        print(f"Local LLM generation failed: {e}")

# --- Test the Pipeline ---
if __name__ == "__main__":
    # Simulating a log that just triggered our FastAPI endpoint
    test_attack_log = {
        "src_ip": "192.168.1.15",
        "dst_ip": "10.138.69.97",
        "bytes_in": 32050,  
        "bytes_out": 450,
        "dst_port": 443,
        "protocol": "HTTPS"
    }
    test_mse = 14.85 
    
    generate_security_alert_local(test_attack_log, test_mse)