from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

# 1. 모델 & 토크나이저 불러오기 (8bit 로딩)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, load_in_8bit=True, device_map="auto"
)

# 2. LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# 3. 데이터셋 로딩
dataset = load_dataset("json", data_files="processed_chat.jsonl", split="train")


# 4. 토큰화 함수
def tokenize(example):
    prompt = f"{example['instruction']}\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(tokenize, batched=True)

# 5. 학습 파라미터
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# 6. 데이터콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. Trainer 구성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. 학습 시작
trainer.train()

# 9. 최종 저장
trainer.save_model("./outputs/final-model")
