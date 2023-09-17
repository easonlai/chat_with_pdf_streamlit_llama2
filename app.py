# Import necessary libraries.
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain

# Set web page title and icon.
st.set_page_config(
    page_title="Chat with PDF",
    page_icon=":robot:"
)

# Set web page title and markdown.
st.title('💬 Chat with PDF 📄 (Powered by Llama 2 🦙🦙)')
st.markdown(
    """
    This is the demonstration of a chatbot with PDF with Llama 2, Chroma, and Streamlit.
    I read the book Machine Learning Yearning by Andrew Ng. Please ask me any questions about this book.
    """
)

# Define a function to get user input.
def get_input_text():
    input_text = st.text_input("Ask a question about your PDF:")
    return input_text

# Define to variables to use "sentence-transformers/all-MiniLM-L6-v2" embedding model from HuggingFace.
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define the Chroma vector store and function to generate embeddings.
db = Chroma(persist_directory="./chroma_db/", embedding_function=embeddings)

# Get user input.
user_input = get_input_text()

# Initialize the Azure OpenAI ChatGPT model.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the path of the Llamaccp model.
model_path = "/Users/easonlai/.cache/huggingface/hub/models--TheBloke--Llama-2-13B-chat-GGUF/snapshots/245bc5104d85dcc9a11a0e0a9ae6de38dfae536f/llama-2-13b-chat.Q4_K_M.gguf"

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Initialize the llamaCpp model.
llm = LlamaCpp(
    model_path=model_path,
    max_tokens=256,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    n_ctx=2048,
    verbose=False,
)

# Define the function to get the response.
if user_input:
    # Perform similarity search for the user input.
    docs = db.similarity_search(user_input)

    # Load the question answering chain.
    chain = load_qa_chain(llm, chain_type="stuff")

    # Get the response from llamaCpp model.
    response = chain.run(input_documents=docs, question=user_input)

    # Display the response.
    st.write(response)
