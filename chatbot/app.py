import streamlit as st
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Define your Google API key here
google_api_key = "AIzaSyDfnYQykoNClC9i-vfLnubOdgymlpDr2-I"

def extract_text_from_pdf(uploaded_file):
    text = ""
    with uploaded_file as pdf_file:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def main():
    st.title("Question Answering App")

    # File uploader for PDF documents
    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        pdf_texts = []
        for uploaded_file in uploaded_files:
            pdf_text = extract_text_from_pdf(uploaded_file)
            pdf_texts.append(pdf_text)
        context = "\n".join(pdf_texts)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(context)

        # Generate embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

        # Define prompt template
        prompt_template = PromptTemplate(template="""
        Answer the question as detailed as possible from the provided context, 
        make sure to provide all the details. If the answer is not in the provided context, 
        just say, "answer is not available in the context," don't provide the wrong answer
        
        Context:
        {context}?
        
        Question:
        {question}
        
        Answer:
        """, input_variables=["context", "question"])

        # Ask question and handle call request
        user_input = st.text_input("Enter your question or type 'call me' to request a call:")
        if user_input.lower() == "call me":
            st.write("Please provide your contact information:")
            user_name = st.text_input("Name")
            user_phone = st.text_input("Phone Number")
            user_email = st.text_input("Email")
            
            if st.button("Submit"):
                if user_name and user_phone and user_email:
                    # You can use the collected user information to initiate a call
                    st.write(f"Calling {user_name} at {user_phone}...")
        else:
            question = user_input
            if st.button("Get Answer"):
                docs = vector_index.get_relevant_documents(question)

                # Initialize generative AI model
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

                # Generate response
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

                # Format the response
                formatted_response = f"The answer is: {response['output_text']}"

                # Display response
                st.write("Formatted Answer:", formatted_response)

if __name__ == "__main__":
    main()
