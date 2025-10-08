from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = 'medicalbot'

# embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OllamaLLM(model="llama3.1", temperature=0.4)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#Tabnine: Edit|Test|Explain|Document}Ask
@app.route("/")
def index():
    return render_template('chat.html')


#Tabnine: Edit|Test|Explain|Document}Ask
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I cannot answer that right now.")
    except Exception as e:
        print("Error in RAG chain:", e)
        answer = "Sorry, I cannot answer that right now."
    
    return jsonify({"answer": answer})


if __name__=='__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)