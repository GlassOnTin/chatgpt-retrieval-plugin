import requests
import os
import json
LENGTH=1000
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
url = "https://retrieval-plugin-server-se2t.onrender.com/upsert"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {BEARER_TOKEN}"
}
data = {
    "documents": [
        {
            "text": ''.join(str(i % 10) for i in range(LENGTH)),
            "metadata": {
                "title": f"{LENGTH} digits",
                "type": "test",
                "source": "dev",
                "status": "testing"
            }
        }
    ]
}
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.status_code)
print(response.text)
