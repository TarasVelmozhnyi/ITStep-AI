# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 1
# Завдання 1
# Прочитайте файл data\lesson9\return_policy.txt Та
# напишіть простий чат бот для відповідей на питання
# користувачів стосовно повернення товару. Діалог завершується
# коли користувач вводить порожній рядок.
# Передавайте усю історію спілкування у форматі:
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI


import os
import dotenv
from langchain_google_genai import GoogleGenerativeAI

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key,
    temperature=0.7,
)

with open('data/lesson9/return_policy.txt', 'r', encoding='utf-8') as file:
    policy = file.read()

chat_history = [f"Instruction: Ти - консультант магазину. Політика повернення: {policy}"]

print("Консультант з повернення товарів")
print("Введіть питання або порожній рядок для виходу")

while True:
    user_input = input("\nВаше питання: ").strip()

    if user_input == "":
        break

    chat_history.append(f"Human: {user_input}")

    response = llm.invoke("\n".join(chat_history))

    chat_history.append(f"AI: {response}")

    print(f"Відповідь: {response}")

print("\n=== Вся розмова ===")
for line in chat_history:
    print(line)