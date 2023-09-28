import pandas as pd
import json

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config

with open('places.json', "r") as file:
    documents = json.load(file)

#создаем из наших документов датафрейм
df = pd.DataFrame(documents)

loader = DataFrameLoader(df, page_content_column='question')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# задаем векторайзер
embeddings = OpenAIEmbeddings(openai_api_key=config.API_KEY_OPENAI)

# создаем хранилище
db = FAISS.from_documents(texts, embeddings)
db.as_retriever()

# создаем цепочку
qa_chain = RetrievalQA.from_chain_type(
llm=OpenAI(temperature=0, openai_api_key=config.API_KEY_OPENAI),
chain_type='stuff',
retriever=db.as_retriever()
)

prompt_template = """Используй контекст для ответа на вопрос, пользуясь следующими правилами:

Не изменяй текст, который находится в кавычках.
В конце обязательно добавь ссылку на полный документ
{answer}
url: {url}
"""

PROMPT = PromptTemplate(
template=prompt_template, input_variables=['answer', 'url']
)

# цепочка с кастомным промтом
chain = LLMChain(
llm=OpenAI(temperature=0, openai_api_key=config.API_KEY_OPENAI, max_tokens=500),
prompt=PROMPT)

relevants = db.similarity_search('не знаю как изменить пароль')
doc = relevants[0].dict()['metadata']

print(chain.run(doc))