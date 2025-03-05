import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


Embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
st.title('Conversational RAG with PDF uploads and chat history')
st.write('Upload PDF"s and chat with their content')
api_key=st.text_input('Enter your Groq API key:',type='password')
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name='Gemma2-9b-It')

    session_id=st.text_input('Session ID',value='default_session')
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose A PDF file",type='pdf',accept_multiple_files=True)

    #Process uploded PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f'./temp.pdf'
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
    # split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        splits=text_splitter.split_documents(documents)
        chroma_path='E:/AI_projects/langchain/1-longchain/RAG_model/OpenAI_ChatBot/RAG_Model_for_PDF"s/Chromadb'
        vector_store=Chroma.from_documents(documents=splits,embedding=Embeddings,persist_directory=chroma_path)
        retriever=vector_store.as_retriever()

        Contextulize_q_system_prompt=(
        'Given a chat history and the latest user question'
        'which might reference context in the chat history'
        'formulate a standalone question which can be understood'
        'without the chat history. Do not answer the question'
        'Just reformulate it if needed and otherwise return it as is.'
    )
        Contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',Contextulize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('user','{input}'),
        ]
    )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,Contextualize_q_prompt)

    # Answer Question
        system_prompt=(
        'you are assistant for question-answering tasks. '
        'Use the following pieces of retrieved context to answer'
        'The question. If you don"t know the answer, say that you'
        'don"t know. Use three sentences maximum and keep the '
        'Answer concise'
        '\n\n'
        '{context}'
    )
        qa_prompt=ChatPromptTemplate.from_messages(
        [('system',system_prompt),
         MessagesPlaceholder('chat_history'),
         ('user','{input}'),
        ]
    )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )
        user_input=st.text_input('your question:')
        if st.button("Generate Response"):
            if user_input:
                session_history=get_session_history(session_id)
                response=conversational_rag_chain.invoke(
                    {'input': user_input},
                    config={
                        'configurable':{'session_id':session_id}
                    },
                )
                # st.write(st.session_state.store)
                st.write('Assistant',response['answer'])
                # st.write('chat History',session_history.messages)
            else:
                st.warning('Kindly enter your queary')
else:
    st.warning('enter Groq API key')
