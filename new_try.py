import requests

API_KEY = "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"  # Replace with your actual key
MODEL = "deepseek/deepseek-chat"  # You can also try: "mistralai/mixtral-8x7b-instruct" or "google/gemma-2b"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://openrouter.ai/",
    "X-Title": "Abhinandan-RAG-App"
}

def ask_api(prompt):
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    resp = requests.post(ENDPOINT, headers=headers, json=data)

    if resp.status_code != 200:
        print("❌ Error:", resp.status_code, resp.text)
        return "Error: API call failed."

    ans = resp.json()
    return ans["choices"][0]["message"]["content"]

if __name__ == "__main__":
    prompt = "hi"
    print(ask_api(prompt))
