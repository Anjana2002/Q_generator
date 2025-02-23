#!/usr/bin/env python
# coding: utf-8

# In[2]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore


# In[3]:


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
)


# In[4]:


reader = SimpleDirectoryReader(input_dir="./question_papers/", recursive=True)
documents = reader.load_data()


# In[6]:


from chromadb import PersistentClient
index = VectorStoreIndex.from_documents(documents)
# index.storage_context.persist('storage') 
chroma_client = PersistentClient(path="storage")
vector_store = ChromaVectorStore(chroma_collection=chroma_client.get_or_create_collection("my_collection"))


# In[7]:


# storage_context = StorageContext.from_defaults(persist_dir="storage")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = load_index_from_storage(storage_context)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
storage_context.persist()


# In[8]:


query_engine = index.as_query_engine()


# In[ ]:


question_paper_prompt = """  
Generate 5 questions from different sections  
"""  

response = query_engine.query(question_paper_prompt)
print(response)


# In[ ]:




