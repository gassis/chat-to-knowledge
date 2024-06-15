# import dependencies
import os
from openai import OpenAI
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from neo4j import GraphDatabase
from langchain_community.graphs.neo4j_graph import Neo4jGraph

# Global constants
NEO4J_URI = st.secrets['NEO4J_URI']
NEO4J_DATABASE = st.secrets['NEO4J_DATABASE']
NEO4J_USERNAME= st.secrets['NEO4J_USERNAME']
NEO4J_PASSWORD= st.secrets['NEO4J_PASSWORD']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

VECTOR_INDEX_NAME = "embedingIndex"
VECTOR_NODE_LABEL = "Chunk"
VECTOR_NODE_PROPERTY = "text"
VECTOR_EMBEDDING_PROPERTY = "embedding"

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

#embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


def configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response based on vector search and retrieval of structured chunks

    general_system_template = """Use as seguintes partes do contexto para responder à pergunta no final.
    Suas respostas deve ser completas e estruturadas. Utilize prioritariamente informações sobre o Instituto 
    Federal de Educação, Ciência e Tecnologia de Goiás ou IFG.
    Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.

    {summaries}

    Question: {question}

    Helpful Answer:"""
    general_user_template = " Question: {question}"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Initialise Neo4j as Vector + Knowledge Graph store
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database='neo4j',  # neo4j by default
        index_name=VECTOR_INDEX_NAME,  # vector index name
        node_label=VECTOR_NODE_LABEL,  # embedding node label
        embedding_node_property=VECTOR_EMBEDDING_PROPERTY,  # embedding value property
        text_node_property=VECTOR_NODE_PROPERTY,  # text by default
        retrieval_query="""
              WITH node AS embNode, score ORDER BY score DESC LIMIT 10
              MATCH (embNode) -[:CHUNK_OF]-> (part) -[*]-> (document:Document) WHERE embNode.embedding IS NOT NULL
              OPTIONAL MATCH (part)-[:SECTION_OF]->(chapter:Chapter),(part)-[:CHAPTER_OF]->(document:Document),(part)-[:CHAPTER_OF]->(title:Title)
              WITH document, part, embNode, score ORDER BY part.title, embNode.chunk_seq_id ASC
              WITH document, part, collect(embNode) AS answers, max(score) AS maxScore
              RETURN {source: document.url, part: part.name, matched_chunk_id: id(answers[0])} AS metadata,
                  reduce(text = "", x IN answers | text + x.text + '.') AS text, maxScore AS score LIMIT 10;
        """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 3}),
        reduce_k_below_max_tokens=False,
        return_source_documents=True,
    )
    return kg_qa

rag_chain = configure_qa_structure_rag_chain(llm, embeddings, embeddings_store_url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

st.title("Chat-to-Knowledge")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# Function for generating LLM response
def generate_response(prompt_input):
    input_data = {"question": prompt_input}
    result = rag_chain.invoke(input_data)
    return result['answer']

if prompt := st.chat_input("O que deseja saber?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate_response(prompt)
        response = st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
