from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

import json
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

pc = Pinecone(api_key=pinecone_api_key)
index_name = "practice1"  # назва бази даних

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


# документ1 -- future of ai
with open('data/lesson_rag/files/future_of_ai.txt','r', encoding='utf-8') as f:
    text_1 = f.read()

doc1 = Document(
    page_content=text_1, #вміст документа
    metadata={'path': 'data/lesson_rag/files/future_of_ai.txt'
    }
)

# документ2 -- intro
with open('data/lesson_rag/files/intro.txt','r', encoding='utf-8') as f:
    text_2 = f.read()

doc2 = Document(
    page_content=text_2,
    metadata={
        "path":"data/lesson_rag/files/intro.txt"
    }
)

# документ3 -- machine_learning
with open('data/lesson_rag/files/machine_learning.txt','r', encoding='utf-8') as f:
    text_3 = f.read()

doc3 = Document(
    page_content=text_3,
    metadata={
        "path":'data/lesson_rag/files/machine_learning.txt'
    }
)

#doc4 -- neural_networks
with open('data/lesson_rag/files/neural_networks.txt','r', encoding='utf-8') as f:
    text_4 = f.read()

doc4 = Document(
    page_content=text_4,
    metadata={
        "path":'data/lesson_rag/files/neural_networks.txt'
    }
)


# documents list
docs = [doc1, doc2, doc3, doc4]

#creating unique documents id
ids = [str(uuid4()) for _ in  range(len(docs))]

id_map = {

}

for doc,id  in  zip(docs, ids):
    id_map[doc.metadata["path"]] = id

print(id_map)

with open ('ids.json', 'w') as f:
    json.dump(id_map, f, indent=2)

# #завантаження документів у базу Данних
vector_store.add_documents(
    documents=docs,
    ids=ids
)




# створення агента
agent = create_react_agent(
    model=llm,  # мовна модель
    tools=[search_doc]
)

# історія повідомлень + інструкції

messages = [
    SystemMessage(
        """
        Ти ввічлий чат-бот. Твоя задача давати інформативні та чіткі відповіді
        на запити користувача.

        У тебе є доступ до таких інструментів:
        * search_doc -- цукає інформацію в базі даних що містить:
            * інформація про суп
            * інформація про здоров'я
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

    # виведемння всієї історії
    print()
    print("Історія")

    for message in messages:
        print(repr(message))

