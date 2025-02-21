#!/usr/bin/env python
# coding: utf-8

# In[1]:


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM  
import os


# In[2]:


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = HuggingFaceLLM(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",  
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
)


# In[3]:


DATA_DIR = "./question_papers/"
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    print("Error: No PDFs found in", DATA_DIR)
    exit()


# In[4]:


reader = SimpleDirectoryReader(input_dir=DATA_DIR)
documents = reader.load_data()

INDEX_DIR = "storage"


# In[5]:


if os.path.exists(INDEX_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    print("Loaded existing index.")
else: 
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()  
    print("Created new index and saved.")


# In[6]:


query_engine = index.as_query_engine()

# Exam Question Paper Prompt
question_paper_prompt = """  
You are an expert in academic exam paper generation. Given previous years' question papers and notes,  
generate a new set of questions following the same format.  
Ensure the difficulty distribution is similar and avoid direct repetition.  

Format:
1. Section A - Short answer questions - 3 questions (5 marks each)  
2. Section B - Medium answer questions - 3 questions (10 marks each)  
3. Section C - Long answer questions - 2 questions (15 marks each)  
"""


# In[7]:


response = query_engine.query(question_paper_prompt)
print("\nGenerated Question Paper:\n", response)


# In[11]:




