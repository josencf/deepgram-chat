# from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# import os

# loader1 = TextLoader('file.txt')

# index = VectorstoreIndexCreator(
#     # vectorstore_cls=Chroma,
#     # embedding=OpenAIEmbeddings(),
#     # text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# ).from_loaders([loader1])

# index.query('What are some Summarization use cases')



from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

from langchain.document_loaders import TextLoader

loader = TextLoader("file.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
query = "What are some use cases for Summarization?"
result = qa.run(query)
print(result)
