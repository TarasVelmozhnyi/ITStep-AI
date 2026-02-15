import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# заголовок
st.title("🇬🇧 Помічник з англійської")

# завантаження апі ключа
api_key = st.secrets.get("GEMINI_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key,
)

user_query = st.chat_input("Введіть слово або фразу...")

# якщо це початок - створити історію
if user_query is None:
    st.session_state['history'] = [
        SystemMessage(
            """
            Ти помічник з вивчення англійської мови.
            Якщо користувач просить перекласти слово/фразу:
            - дай переклад
            - наведи приклад використання в реченні
            Відповідай українською.
            """
        )
    ]

# якщо повідомлення введено
if user_query:
    human_message = HumanMessage(user_query)
    st.session_state['history'].append(human_message)
    response = llm.invoke(st.session_state['history'])
    st.session_state['history'].append(response)

# вивести всю історію
for message in st.session_state['history']:
    if isinstance(message, SystemMessage):
        continue

    role = "human" if isinstance(message, HumanMessage) else "ai"

    with st.chat_message(role):
        st.markdown(message.content)