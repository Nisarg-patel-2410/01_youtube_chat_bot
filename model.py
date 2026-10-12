from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace  
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os 
import sys
import dotenv


dotenv.load_dotenv()

URL = input("Give me url of video:\n")
video_id = URL.split("?v=")[1].split("&")[0]
FILE_PATH = f"{video_id}.txt".replace("-", "")
print(video_id)
ytt_api = YouTubeTranscriptApi()

print("Extracting Captions")
try:
    script = ytt_api.fetch(video_id)
except:
    print("No caption available")
    sys.exit()
    
if os.path.exists(f".\\{FILE_PATH}"):
    # print("exists")
    pass
else:
    for i in script:
        with open(FILE_PATH, "+a") as f:
            f.write(i.text)
            f.write("\n")

print("Extracting Captions: Completed ")
print("Loading documents")
loder = TextLoader(FILE_PATH)
docs = loder.load()
print("Loading documents: Completed")

# print(docs)

print("Spiting documents")
spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = spliter.split_documents(docs)

# chunks[0]
texts = [chunk.page_content for chunk in chunks]
print("spliting documents: Completed")

# print(texts[0])
# for i in chunks:
#     print(i)
#     print()

print("Generating embedding of documents")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Generating embedding of documents: Completed")

FOLDER_PATH = f"{video_id}".replace("-", "")

if os.path.exists(f"youtube_vectorstore\\{FOLDER_PATH}"):
    vector_store = FAISS.load_local(
        f".\\youtube_vectorstore\\{FOLDER_PATH}",
        embeddings,
        allow_dangerous_deserialization=True
    )
else: 
    print("New")
    vector_store = FAISS.from_documents(chunks,embeddings)

    vector_store.save_local(f"youtube_vectorstore\\{FOLDER_PATH}")

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3},
    search_type = "mmr"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# query = input("Ask something about the video: ")
# results = vector_store.similarity_search(
#     query,
#     k=5
# )

prompt = ChatPromptTemplate.from_template("""
You are a question-answering assistant for a video transcript.

Answer the question using ONLY the information in the provided context.

Rules:
1. Give a direct and concise answer.
2. Use the context as your ONLY source of information.
3. Do not discuss the context itself.
4. Do not say phrases like "the context says", "the video is about", or "the answer is".
5. Do not use outside knowledge.
6. Do not invent or assume information.
7. If the context does not contain enough information to answer the question, say exactly:
"I don't have enough information from the video to answer that."
8. Do not give redundent ans

Context:
{context}

Question:
{question}

Answer:
""")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    temperature=0.2,
    provider="featherless-ai"
)

llm = ChatHuggingFace(llm=llm)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    |prompt
    |llm   
    |StrOutputParser()     
)

while True:
    query = input("Ask question about video:\n")
    if query.lower() == "exit":
        break
    
    response = chain.invoke(query)
    print(response)
    print("\n\n")

    docs = retriever.invoke(query)

    for i, doc in enumerate(docs):
        print(f"\n--- CHUNK {i+1} ---")
        print(doc.page_content)
    print("\n\n\n")
