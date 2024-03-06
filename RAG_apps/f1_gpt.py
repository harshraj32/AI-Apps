from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import hashlib
import json
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import numpy as np
from renumics import spotlight
from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
docs_vectorstore = Chroma(
    collection_name="docs_store",
    embedding_function=embeddings_model,
    persist_directory="docs-db",
)

loader = DirectoryLoader(
    "docs",
    glob="*.html",
    loader_cls=BSHTMLLoader,
    loader_kwargs={"open_encoding": "utf-8"},
    recursive=True,
    show_progress=True,
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)



def stable_hash(doc: Document) -> str:
    """
    Stable hash document based on its metadata.
    """
    return hashlib.sha1(json.dumps(doc.metadata, sort_keys=True).encode()).hexdigest()

split_ids = list(map(stable_hash, splits))
docs_vectorstore.add_documents(splits, ids=split_ids)
docs_vectorstore.persist()



llm = ChatOpenAI(model="gpt-4", temperature=0.0)
retriever = docs_vectorstore.as_retriever(search_kwargs={"k": 20})



template = """
You are an assistant for question-answering tasks.
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{source_documents}
=========
FINAL ANSWER: """
prompt = ChatPromptTemplate.from_template(template)




def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata['source']}" for doc in docs
    )


rag_chain_from_docs = (
    RunnablePassthrough.assign(
        source_documents=(lambda x: format_docs(x["source_documents"]))
    )
    | prompt
    | llm
    | StrOutputParser()
)
rag_chain = RunnableParallel(
    {
        "source_documents": retriever,
        "question": RunnablePassthrough(),
    }
).assign(answer=rag_chain_from_docs)


question = "Who built the nuerburgring"
response = rag_chain.invoke(question)
response["answer"]



response = docs_vectorstore.get(include=["metadatas", "documents", "embeddings"])
df = pd.DataFrame(
    {
        "id": response["ids"],
        "source": [metadata.get("source") for metadata in response["metadatas"]],
        "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
        "document": response["documents"],
        "embedding": response["embeddings"],
    }
)
df["contains_answer"] = df["document"].apply(lambda x: "Eichler" in x)
df["contains_answer"].to_numpy().nonzero()



question_row = pd.DataFrame(
    {
        "id": "question",
        "question": question,
        "embedding": embeddings_model.embed_query(question),
    }
)
answer_row = pd.DataFrame(
    {
        "id": "answer",
        "answer": answer,
        "embedding": embeddings_model.embed_query(answer),
    }
)
df = pd.concat([question_row, answer_row, df])



question_embedding = embeddings_model.embed_query(question)
df["dist"] = df.apply(
    lambda row: np.linalg.norm(
        np.array(row["embedding"]) - question_embedding
    ),
    axis=1,
)



spotlight.show(df)