import requests
import os
import json
LENGTH=1000
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
url = "https://retrieval-plugin-server-se2t.onrender.com/"
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
response = requests.post(url + "upsert", headers=headers, data=json.dumps(data))
print(response.status_code)
print(response.text)

# Delete All!
import requests
import os
import json
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
url = "https://retrieval-plugin-server-se2t.onrender.com/"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {BEARER_TOKEN}"
}
data_del = { 
    "delete_all": True
}
response = requests.post(url + "delete", headers=headers, data=json.dumps(data_del))
print(response.status_code)
print(response.text)
