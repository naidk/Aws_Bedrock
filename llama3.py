import boto3
import json

prompt_data = "Act as Shakespeare and write a poem on Machine Learning."

# Bedrock runtime client
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Properly insert your prompt into the model input format
payload = {
    "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt_data}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)

model_id = "meta.llama3-70b-instruct-v1:0"

# Call the model
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Extract output
response_body = json.loads(response.get("body").read())
print(response_body["generation"])
