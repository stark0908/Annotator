import json

with open("instances_default.json", "r") as f:
    data = json.load(f)

with open("instances_default.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
