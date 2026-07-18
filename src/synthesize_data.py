import os
import csv
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the environment variables (GEMINI_API_KEY)
load_dotenv()

# 2. Initialize the GenAI client
client = genai.Client()

def generate_normal_traffic(num_batches=10, rows_per_batch=100):
    output_file = 'data/synthetic/normal_traffic.csv'
    
    # Target CloudWatch Schema
    headers = [
        "bytes_in", "bytes_out", "creation_time", "end_time", "src_ip", 
        "src_ip_country_code", "protocol", "response.code", "dst_port", 
        "dst_ip", "rule_names", "observation_name", "source.meta", 
        "source.name", "time", "detection_types"
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(num_batches):
            print(f"Generating batch {i+1} of {num_batches}...")
            
            prompt = f"""
            Generate {rows_per_batch} rows of normal web traffic data matching this exact schema:
            {','.join(headers)}
            
            Constraints for Normal Traffic:
            1. bytes_in must be relatively small (100 to 2000 bytes) representing standard GET/POST requests.
            2. bytes_out must be larger (3000 to 100000 bytes) representing served content.
            3. response.code should be primarily 200, with occasional 301, 302, or 404.
            4. rule_names, observation_name, and detection_types MUST strictly be "None".
            5. dst_port is 443 and protocol is HTTPS.
            6. Timestamps should be in ISO 8601 format (e.g., 2024-04-25T23:00:00Z) incrementing logically.
            7. Output ONLY CSV rows. NO headers, NO markdown formatting, NO explanations.
            """
            
            # --- Exponential Backoff Retry Logic ---
            max_retries = 5
            base_delay = 15
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model='gemini-3.5-flash',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a data engineer generating synthetic benign network logs.",
                            temperature=0.7,
                        )
                    )
                    
                    # Clean up potential markdown formatting from the response
                    csv_data = response.text.replace("```csv", "").replace("```", "").strip()
                    
                    for row in csv_data.split('\n'):
                        if row.strip():
                            f.write(row.strip() + '\n')
                            
                    print(f"Batch {i+1} saved successfully.")
                    break  # Success! Break out of the retry loop.
                    
                except Exception as e:
                    print(f"Error during generation (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # Exponential backoff: 15s, 30s, 60s, 120s...
                        sleep_time = base_delay * (2 ** attempt)
                        print(f"Retrying in {sleep_time} seconds...\n")
                        time.sleep(sleep_time)
                    else:
                        print(f"Failed to generate batch {i+1} after {max_retries} attempts. Moving to next batch.")
            
            # Standard Rate Limit Protection between batches
            if i < num_batches - 1:
                print("Sleeping for 15 seconds to respect API rate limits...\n")
                time.sleep(15)

if __name__ == "__main__":
    print("Starting synthetic data generation...")
    generate_normal_traffic()
    print("Generation complete. Data saved to data/synthetic/normal_traffic.csv")