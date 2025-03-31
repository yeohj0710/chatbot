from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# 1. 모델 경로 지정
base_model = "mistralai/Mistral-7B-v0.1"
lora_model = "./outputs/final-model"  # LoRA 학습된 모델 저장 위치

# 2. 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, lora_model)

# 3. 채팅 파이프라인 구성
chat = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 4. 대화 루프
print("💬 Chatbot 시작! (종료하려면 'exit' 입력)\n")
history = []

while True:
    user_input = input("👤 You: ")
    if user_input.lower() == "exit":
        break

    # 간단한 히스토리 포함 프롬프트 구성
    prompt = ""
    for line in history[-5:]:
        prompt += line + "\n"
    prompt += f"유저: {user_input}\n챗봇:"

    response = chat(
        prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1
    )[0]["generated_text"]

    # 응답만 추출
    bot_reply = response.split("챗봇:")[-1].strip().split("\n")[0]
    print(f"🤖 Bot: {bot_reply}")
    history.append(f"유저: {user_input}")
    history.append(f"챗봇: {bot_reply}")
