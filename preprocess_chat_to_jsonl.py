import re
import json

lines = open("chat.txt", encoding="utf-8").read().splitlines()
messages = []

msg_pattern = re.compile(r"^\d{4}년.*?, (.+?) : (.+)$")

for line in lines:
    match = msg_pattern.match(line)
    if match:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        messages.append(f"{speaker}: {text}")

pairs = []
for i in range(len(messages) - 1):
    pairs.append({"instruction": messages[i], "input": "", "output": messages[i + 1]})

with open("processed_chat.jsonl", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")
