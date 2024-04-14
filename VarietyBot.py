import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


st.set_page_config(
    page_title="ChatCUD",
    page_icon="💬",
)

page_config = {
    st.markdown(
    "<h1 style='text-align: center; color: #b22222; font-family: Arial, sans-serif; background-color: #292f4598;'>chatCUD 💬</h1>",
    unsafe_allow_html=True
    ),
    st.markdown("<h4 style='text-align: center; color: white; font-size: 20px; animation: bounce-and-pulse 60s infinite;'>Your CUD AI Assistant</h4>", unsafe_allow_html=True),
}

gemini_config = {'temperature': 0.7, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048}
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-pro", generation_config=gemini_config)

import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import time

gemini_config = {'temperature': 0.8, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 2048}
page_config = {st.title('🤖🌐 VarietyBot'),
st.caption("Please ensure clarity in your questions for a smooth conversation. If you've uploaded a PDF, just mention 'my pdf' in your questions. Otherwise, ask usual questions for AI-Generated answers ☺")
}
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel(model_name="models/gemini-pro",generation_config=gemini_config)

# Function to extract text from uploaded PDF files
def extract_text(upload):
    # Initialize an empty string to store extracted text
    pdf_text = ''
    # Loop through each uploaded PDF file
    for pdf in upload:
        # Read the PDF file
        read_pdf = PdfReader(pdf)
        # Loop through each page in the PDF file
        for page in read_pdf.pages:
            # Extract text from the current page and append it to pdf_text
            pdf_text += page.extract_text()
    # Return the extracted text
    return pdf_text

# Function to split text into smaller chunks
def get_chunks(text):
    # Initialize a text splitter object with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=9000, chunk_overlap=900)
    # Split the text into chunks using the text splitter
    chunks = text_splitter.split_text(text)
    # Return the chunks
    return chunks

# Function to generate embeddings from text chunks and store them
def get_embeddings_and_store_pdf(chunk_text):
    # Initialize a Google Generative AI Embeddings model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create embeddings from the chunk text
    create_embedding = FAISS.from_texts(chunk_text, embedding=embedding_model)
    # Save the created embeddings locally
    create_embedding.save_local("embeddings_index")

# Function to get user input and generate responses based on similarity to stored PDFs
def get_generated_user_input(user_question):
    # Initialize a Google Generative AI Embeddings model
    text_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load stored embeddings from local storage
    stored_embeddings = FAISS.load_local("embeddings_index", text_embedding, allow_dangerous_deserialization=True)
    # Perform similarity search based on user question
    check_pdf_similarity = stored_embeddings.similarity_search(user_question)

    # Define a prompt template for generating responses
    my_prompt = '''
    Answer the following question with the given context:
    Context:\n{context}?\n
    Question:\n{question}\n
    '''
    # Initialize a Google Generative AI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # Initialize a prompt template with the defined prompt
    prompt_template = PromptTemplate(template=my_prompt, input_variables=["context", "question"])
    # Load a question answering conversation chain
    conversation_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)
    # Generate response based on user question and similarity search results
    response = conversation_chain({"input_documents": check_pdf_similarity, "question": user_question}, return_only_outputs=True)
    # Return the generated response text
    return response['output_text']

# Function to handle user responses
def user_response(user_question):
    # Generate a response based on the user question
    generated_prompt = get_generated_user_input(user_question)
    # Construct a prompt for the user interaction
    prompt = f"You are a helpful AI assistant at the Canadian University Dubai, this is the information given based on the user question return this but make it sound better\n{generated_prompt}?\n do not be irrelevant\nQuestion: \n{user_question}"
    # Send the prompt and get the response
    response = st.session_state.chat_history.send_message(prompt)
    # Return the response text
    return response.text

# Function to clear chat conversation history
def clear_chat_convo():
    # Clear the chat history
    st.session_state.chat_history.history = []

# Function to determine role-based icons
def role_name(role):
    # Assign icons based on the role
    if role == "model":
        return "bot.png"
    elif role == 'user':
        return 'user.png'
    else:
        return None 

# Function to stream text responses
def stream(response):
    # Iterate through the text response, yielding each word with a slight delay
    for word in response.text.split(" "):
        yield word + " "
        time.sleep(0.04)

# Extracts the user question from pdf prompt in get_generated_user_input() 
def extract_user_question(prompt_response):
    # Iterate through the parts of the prompt response in reverse order
    for part in reversed(prompt_response):
        # Check if the part contains the keyword "Question:"
        if "Question:" in part.text:
            # Split the text after "Question:" and return the extracted user question
            return part.text.split("Question:")[1].strip()


def main():
    with open('dark.css') as f:
        # Apply the CSS style to the page
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 

    start_conversation = model.start_chat(history=[])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = start_conversation
    
    for message in st.session_state.chat_history.history:
        # Get the role name of the message and fetch corresponding avatar if available
        avatar = role_name(message.role)
        # Check if avatar exists
        if avatar:
            # Display the message with the role's avatar
            with st.chat_message(message.role, avatar=avatar):
                # Check if the message has 'content' in its parts
                if "Canadian University Dubai" in message.parts[0].text:
                    # Extract the user's question from the message parts (if available)
                    user_question = extract_user_question(message.parts)
                    # Check if a user question is extracted
                    if user_question:
                        # Display the user question using Markdown
                        st.markdown(user_question)
                else:
                    # If 'content' is not found in the parts, display the message text using Markdown
                    st.markdown(message.parts[0].text)

    
    st.sidebar.markdown("<div style='display: flex; justify-content: center;'><h3>Choose One To Proceed</h3></div>", unsafe_allow_html=True)
    with st.sidebar:
        st.sidebar.markdown("<div style='display: flex; justify-content: center;'><h3>Chat PDF File <h3></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload One Or More PDF Files", type="pdf",accept_multiple_files=True)
        if uploaded_file is not None:
            if st.sidebar.button("Process PDF File"):
                with st.spinner("Processing..."):
                    try:
                        texts = extract_text(uploaded_file)
                        chunk=get_chunks(texts)
                        get_embeddings_and_store_pdf(chunk)
                        st.success("Proceed to asking PDF")
                    except Exception as e:
                        st.error(f"Error during PDF processing: {e}")
        else:
            st.sidebar.info("Upload PDF to proceed")


    user_question = st.chat_input("Ask chatCUD...")

    if user_question is not None and user_question.strip() != "":
        try: 
            with st.chat_message("user"):
                st.write(user_question)

            response = user_response(user_question)

            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)

        except Exception as e:
            st.error(f"Error handling User Question: {e}")

    st.sidebar.button("Click to Clear Chat History", on_click=clear_chat_convo)
if __name__ == "__main__":
    main()
