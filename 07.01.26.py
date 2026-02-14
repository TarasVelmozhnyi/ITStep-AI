

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

import os
import dotenv
from uuid import uuid4

# завантаження апі ключа
dotenv.load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# модель для кодування текстів(embedding model)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  #  text-embedding-004	вже застаріла і вже як місяць не працює
    google_api_key=gemini_api_key
)

# # кодування текстів
# # отримані числа називають вектор
# vec1 = embeddings.embed_query("Суп корисний при застуді")
#
# print(vec1)
# print(len(vec1))
#
#
# vec2 = embeddings.embed_query("При застуді корисно їсти суп")
# print(vec2)
#
# vec3 = embeddings.embed_query("Бігати більше 10 км шкідливо для здоров'я")
# print(vec3)
#



# створення весторної бази даних
pc = Pinecone(api_key=pinecone_api_key)
index_name = "soup"  # назва бази даних

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,      # кількість чисел при кодування
        metric="cosine",    # формула для схожості
        spec=ServerlessSpec(
            cloud="aws",         # хмарний сервер(амазон)
            region="us-east-1"   # регіон(Каліфорнія)
        ),
    )

index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# створення документів

# документ1 -- Суп корисний при застуді
doc1 = Document(
    page_content="Суп корисний при застуді",   # вміст дукумента
    metadata={               # додаткова інформація
        "type": "здоров'я",
        "author": "Anton Halysh"
    }
)


# документ2 -- Суп придумали в Китаї
doc2 = Document(
    page_content="Суп придумали в Китаї",   # вміст дукумента
    metadata={               # додаткова інформація
        "type": "історія",
        "author": "Anton Halysh",
        "date": "2025 01 07"
    }
)

# документ3 -- Бігати більше 10 км шкідливо для здоров'я
doc3 = Document(
    page_content="Бігати більше 10 км шкідливо для здоров'я",   # вміст дукумента
    metadata={               # додаткова інформація
        "type": "здоров'я",
        "author": "Unknown"
    }
)

# список документів
docs = [doc1, doc2, doc3]

# створення унікальних id для документів
ids = [str(uuid4()) for _ in range(len(docs))]

# # print(ids)

# # завантаження документів у базу даних
#
# vector_store.add_documents(
#     documents=docs,
#     ids=ids
# )


# отримати схожі документи
user_input = "Чи шкідливо бігати більше 10 км?"

result_docs = vector_store.similarity_search(
    user_input,   # текст для порівняння схожості
    k=2,          # кількість документів у відповіді
)

for doc in result_docs:
    print(doc)









































#
# # створення агентів
# # агент -- чат-бот(llm) + інструменти
#
# import os
# import dotenv
# from typing import List
#
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.documents import Document
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import (
#     HumanMessage,
#     AIMessage,
#     SystemMessage,
#     trim_messages, BaseMessage
# )
#
# # завантаження апі ключа
# dotenv.load_dotenv()
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
#
# # створити llm
# llm = ChatGoogleGenerativeAI(
#     model='gemini-2.5-flash',
#     api_key=gemini_api_key,
# )
#
# # модель для кодування текстів(embedding model)
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key=gemini_api_key
# )
#
# # створення весторної бази даних
# pc = Pinecone(api_key=pinecone_api_key)
# index_name = "soup"  # назва бази даних
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=768,      # кількість чисел при кодування
#         metric="cosine",    # формула для схожості
#         spec=ServerlessSpec(
#             cloud="aws",         # хмарний сервер(амазон)
#             region="us-east-1"   # регіон(Каліфорнія)
#         ),
#     )
#
# index = pc.Index(index_name)
# vector_store = PineconeVectorStore(
#     index=index,
#     embedding=embeddings
# )
#
#
# # інструмент -- функція
# # обов'язкова документація
# def search_doc(user_query: str) -> List[Document]:
#     """
#     Шукає схожі документи з релевантної інформацією до запиту користувача
#
#     :param user_query: запит користувача
#     :return: список документів з релевантною інформацією
#     """
#     result_docs = vector_store.similarity_search(
#         user_query,  # текст для порівняння схожості
#         k=2,         # кількість документів у відповіді
#     )
#
#     return result_docs
#
#
# # створення агента
# agent = create_react_agent(
#     model=llm,  # мовна модель
#     tools=[search_doc]
# )
#
# # історія повідомлень + інструкції
#
# messages = [
#     SystemMessage(
#         """
#         Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
#         на запити користувача.
#
#         У тебе є доступ до таких інструментів:
#         * search_doc -- цукає інформацію в базі даних що містить:
#             * інформація про суп
#             * інформація про здоров'я
#         """
#     )
# ]
#
# while True:
#     user_query = input("Ви: ")
#
#     if user_query == '':
#         break
#
#     # переводимо str рядок у  HumanMessage
#     human_message = HumanMessage(user_query)
#
#     # добавляємо повідослення користувача до історії
#     messages.append(human_message)
#
#     # застосування агента
#     # треба передавати словник
#     input_data = {
#         "messages": messages
#     }
#
#     response = agent.invoke(input_data)
#     # response -- словник з усією історією + відповідь моделі
#
#     # отримання всіє історії повідомлень
#     messages = response['messages']
#
#     # отримати фінальну відповідь моделі
#     answear = messages[-1]
#     print(answear.content)
#
#     # виведемння всієї історії
#     print()
#     print("Історія")
#
#     for message in messages:
#         print(repr(message))
