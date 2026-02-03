

import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages, BaseMessage
)


# завантаження апі ключа
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
)

# інструмент для пошуку в інтернеті
search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key, type="places")


def find_restaurants(query: str) -> str:
    """шукає ресторани та повертає інформацію прро них"""

    search_info = search.results(query)
    search_places_info = search_info["places"]
    result_list = []

    for place in search_places_info[:5]:
        result_list.append({
            "name": place.get("title", "Без назви"),
            "website": place.get("website", "Немає"),
            "rating": place.get("rating", "—")
        })

    return result_list


# створення агента
agent = create_react_agent(
    model=llm,  # мовна модель
    tools=[find_restaurants]
)

messages = [
    SystemMessage(
        """Ти - помічник для пошуку ресторанів. 
    Використовуй інструмент SearchRestaurants, щоб знайти ресторани.
    Надавай користувачу інформацію: назву, посилання на сайт та рейтинг.
    Якщо користувач не вказує місто, запитай уточнення.
    
    # Інструменти
    {find_restaurants}
    
        """
    )
]



while True:
    user_query = input("Ви: ")

    if user_query == '':
        break

    # переводимо str рядок у  HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо повідослення користувача до історії
    messages.append(human_message)

    # застосування агента
    # треба передавати словник
    input_data = {
        "messages": messages
    }

    response = agent.invoke(input_data)
    # response -- словник з усією історією + відповідь моделі

    # отримання всіє історії повідомлень
    messages = response['messages']

    # отримати фінальну відповідь моделі
    answear = messages[-1]
    print(answear.content)


