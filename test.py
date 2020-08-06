import requests

json = {"hello": 130}
res = requests.post("http://localhost:5000/predict", json=json)
res = res.json()
print(res["result"])