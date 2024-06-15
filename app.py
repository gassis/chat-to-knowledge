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

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.answer)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
