import requests

url = "http://127.0.0.1:5000/detect"
files = {"image": open("backend/sample.jpeg", "rb")}
response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
