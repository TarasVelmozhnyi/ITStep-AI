import os
from itertools import count

import dotenv
from langchain.chains import conversation
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages,
    BaseMessage, ai
)
from typing import List, Union


# завантаження апі ключа
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=api_key,
)

messages = [SystemMessage(
    content="Ти розумний чат бот та можеш відповісти коротко по суті")
]

trimmer = trim_messages(
    strategy='last',  # залишати останні повідомлення

    token_counter=len,  # рахуємо кількість повідомлень
    max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)

    start_on='human',  # історія завжди починатиметься з HumanMessage
    end_on='human',  # історія завжди закінчуватиметься з HumanMessage
    include_system=True  # SystemMessage не чіпати
)

summary_chain = llm | trimmer

while True:
    user_query = input("Ви: ")
    if user_query == 'exit':
        break

    messages.append(HumanMessage(content=user_query))

    response = llm.invoke(messages)
    print(f"AI: {response.content}")

    # Додаємо відповідь до історії
    messages.append(AIMessage(content=response.content))

    trimmed_messages = trimmer.invoke(messages)

    human_message = HumanMessage(user_query)

    print(f"AI: {response.content}")

    chat_count = len([m for m in messages if isinstance(m, (HumanMessage, AIMessage))])

    if chat_count > 4:
        print("\n[Підсумовую розмову...]")

        history_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_text += f"Людина: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"AI: {msg.content}\n"


promt = PromptTemplate.from_template(
    '''Ти - асистент, який підсумовує розмову. Підсумуй наступну частину розмови в декілька речень, 
    зберігаючи якомога більше деталей та конкретної інформації.
    
    ###Розмова###
    {conversation}
    '''
)

chain = llm | promt

result = summary_chain.invoke({
    'conversation': history_text,
})

print(result.content)
