


# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 6
# Завдання 1
# Добавте в створену базу даних файл
# data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти
# частинами. Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два
# порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його
# назва
#  створіть документи для кожного блоку. В метаданих:
# o назва файлу
# o назва блоку
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента





#  "text-embedding-004"	вжезастаріла і з 14 січня не працює

from langchain_google_genai import ChatGoogleGenerativeAI
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

# створити llm
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=gemini_api_key,
)


# модель для кодування текстів(embedding model)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",  #  "text-embedding-004"	вжезастаріла і з 14 січня не працює
    google_api_key=gemini_api_key
)

pc = Pinecone(api_key=pinecone_api_key)
index_name = "soupe"  # назва бази даних

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


# добавили huge_file.txt


huge_file_path = "data/lesson_rag/huge_file.txt"

# читаємо файл
with open(huge_file_path, "r", encoding="utf-8") as f:
    huge_content = f.read()

document = Document(huge_content)

# розділяємо на блоки (2 порожні рядки)
blocks = huge_content.split("\n\n\n")

huge_docs = []
huge_ids = []

for block in blocks:
    block = block.strip()
    if not block:
        continue

    #назва блоку
    lines = block.split("\n")
    block_title = lines[0].strip()

    # створюємо документ
    doc = Document(
        page_content=block,
        metadata={
            "path": huge_file_path,
            "block_title": block_title
        }
    )

    huge_docs.append(doc)
    huge_ids.append(str(uuid4()))

# даємо в Pinecone
vector_store.add_documents(
    documents=huge_docs,
    ids=huge_ids
)



#  ids.json


# якщо файл існує — зчитуємо старі ID
if os.path.exists("ids.json"):
    with open("ids.json", "r", encoding="utf-8") as f:
        id_map = json.load(f)
else:
    id_map = {}


# додаємо нові ID
for doc, id in zip(huge_docs, huge_ids):
    key = f"{doc.metadata['path']} | {doc.metadata['block_title']}"
    id_map[key] = id

with open("ids.json", "w", encoding="utf-8") as f:
    json.dump(id_map, f, indent=2)

print("huge_file.txt успішно додано частинами ")

