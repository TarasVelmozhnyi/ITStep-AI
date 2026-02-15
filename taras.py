import streamlit as st

#
# за замовчуванням щапускається нескіченний цикл
# Сторінка сайту постійно оновлюється і відповідно
# код нижче постіно запускається

# # заголовок сайту
# st.title("Taras")
#
# # звичайний текст
# st.markdown("Звичайний текст. Можливо опис вашої програми")

# # отримати повідомлення від користувача
# user_query = st.chat_input("Ваше повідомлення")
#
# st.markdown(f"Ви ввели {user_query}")
#
# if user_query == 'Привіт':
#     st.markdown(f"Як справи")
#
#
# # глобальна пам'ять в streamlit
# # session_state -- dict з зміними
#
# if user_query == None:
#     # це самий початок(користувач ще нічого не писав
#     st.session_state['history'] = []
#
# # добавити user_query в історію
# st.session_state['history'].append(user_query)
#
# st.markdown(f"Ви ввели {st.session_state['history']}")
#






# ЧАТ-БОТ

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

# заголовок
st.title("HELLO I'm Taras, твій помічник")

# завантаження апі ключа за допомогою streamlit
api_key = st.secrets.get("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key,
)

user_query = st.chat_input("Ваше повідомлення")

# якщо це початок то створити історію в session state
if user_query is None:
    # історія повідомлень
    st.session_state['history'] = [
        # перше повідомлення з основними інструкціями(промпт)
        SystemMessage(
            """
            Ти -- ввічливий чат бот, твоя задача давити короткі та
            чіткі відповіді на питання
            """
        )
    ]

# якщо повідомлення введено, то дати відповідь від моделі
if user_query:
    # переволимо повідомлення в HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо до історії повідомлень
    st.session_state['history'].append(human_message)

    # запускаємо модель
    response = llm.invoke(st.session_state['history'])

    # response -- AIMessage
    # добавляємо до історії повідомлень
    st.session_state['history'].append(response)


# вивести всю історію спілкування
for message in st.session_state['history']:
    # пропускаємо SystemMessage
    if isinstance(message, SystemMessage):
        continue

    # отримати вміст
    text = message.content

    # отримати роль
    if isinstance(message, HumanMessage):
        role = "human"
    else:
        role = 'ai'

    # вивести повідомлення з підписом
    with st.chat_message(role):
        st.markdown(text)