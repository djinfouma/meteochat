import os
import openai
import time
import json
from openai import AzureOpenAI

# Load secrets from secrets.json
with open('secrets.json') as f:
    secrets = json.load(f)['gpt-4o']

# Extract api_key, endpoint, and version
api_key = secrets['api_key']
endpoint = secrets['endpoint']
api_version = "2023-12-01-preview"

# Initialize the Azure OpenAI client
client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

# Upload the JSONL file
dataset = client.files.create(file=open('general-public_expert.jsonl', 'rb'), purpose='fine-tune')
print("Uploaded file ID:", dataset.id)

# Wait for the file to be processed
while True:
    try:
        print('Waiting for file to be processed...')
        file_handle = client.files.retrieve(dataset.id)  # Retrieve file details

        # Check the file status
        status = file_handle.status
        print("Current file status:", status)

        if status in ['processed', 'succeeded']:
            print('File processed successfully.')
            break
        elif status == 'failed':
            print('File processing failed:', file_handle)
            exit()
    except Exception as e:
        print("Error while checking file status:", e)
        exit()

    time.sleep(3)

# Start the fine-tuning job
try:
    job = client.fine_tuning.jobs.create(training_file=dataset.id, model="gpt-4o-2024-08-06")

    while True:
        print('Waiting for fine-tuning to complete...')
        job_handle = client.fine_tuning.jobs.retrieve(job.id)
        if job_handle.status == 'succeeded':
            print('Fine-tuning completed successfully.')
            print('Fine-tuned model details:', job_handle)
            print('Model ID:', job_handle.fine_tuned_model)
            break
        elif job_handle.status == 'failed':
            print('Fine-tuning failed:', job_handle)
            exit()
        time.sleep(3)
except Exception as e:
    print("Error during fine-tuning:", e)
