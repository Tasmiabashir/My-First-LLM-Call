from transformers import pipeline
print("⏳ Loading the model... (first time takes a few minutes to download)")
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto"  
)
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "What is Machine Learning? Explain in 3 simple sentences."
    }
]
print("\n🚀 Sending your question to the model...\n")
output = pipe(
    messages,
    max_new_tokens=200,    
    do_sample=True,
    temperature=0.7,     
    top_p=0.95
)
response = output[0]["generated_text"][-1]["content"]
print("=" * 50)
print("🤖 MODEL RESPONSE:")
print("=" * 50)
print(response)
print("=" * 50)
print("\n✅ SUCCESS! Your first LLM call is complete!")