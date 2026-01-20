from typing import List
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, ResponseSchema
import dotenv
import os


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    api_key=api_key,
)

class ExerciseItem(BaseModel):
    exercise_tren: str = Field(description="Назва вправи")
    exercise_type: List[str] = Field(description="Тип вправи")

parser = PydanticOutputParser(pydantic_object=ExerciseItem)
instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template(template =
    """
    prompt - ти фітнестренер ,створи програму тренувань під цілі клієнта
    
    ### цілі : 
    {tren}
    
    ### відповідь:
    {instructions}
    
    """,
    partial_variables={"instructions": instructions}
)

chain = prompt | llm | parser

tren = input("який результат ви б хотіли отриммати від тренувань? ")

response = chain.invoke({
    "tren": tren,
})

print({response.exercise_tren})
print("Типи вправ:")

for exercise in response.exercise_type:
    print(f"  - {exercise}")


class TrenPlan(BaseModel):
    exercise_plan: List[str] = Field(description="план тренування")
    exercise_type: str = Field(description="підготовка")
    weekly_hours: int = Field(description="Кількість годин")

parser2 = PydanticOutputParser(pydantic_object=TrenPlan)
instructions = parser2.get_format_instructions()

prompt2 = PromptTemplate.from_template(
    """
    Ти - професійний фітнес-тренер. 
    На основі списку вправ створи персоналізований план тренувань.
    
    ### Є такі данні
    ### Список вправ: план тренування {exercise_plan}
    
    ### Рівень підготовки користувача:{exercise_type}
    
    ### <UNK> <UNK>:{weekly_hours}
    
    ### Відповідь така:
    {instructions}
    """,
    partial_variables={"instructions": instructions}
)


chain2 = prompt2 | llm | parser2


exercise_plan = input("рівен підготовки (низький середній чи професіонал ? ")
weekly_hours = input("cкільки годин на тиждень <UNK> <UNK> <UNK>? ")

print(exercise_plan)
response2 = chain2.invoke({
    "exercise_plan": exercise_plan,
    "weekly_hours": weekly_hours,
    "exercise_type": exercise_plan,
    "instructions": instructions
})

exercise_plan = response2.exercise_plan
print("план тренування")
for i, exercise in enumerate(response2.exercise_plan, 1):
    print(f"{i}. {exercise}")

print(response2.exercise_plan)
print(f"підготовка: {response2.exercise_plan}")

print("Годин на тиждень:", {weekly_hours})
