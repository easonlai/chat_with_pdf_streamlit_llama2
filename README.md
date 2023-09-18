# Semantic Search over Documents (Chat with PDF) with Llama 2 🦙 & Streamlit 🌠

In this repository, you will discover how [Streamlit](https://streamlit.io/), a Python framework for developing interactive data applications, can work seamlessly with the Open-Source Embedding Model ("[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)") in [Hugging Face](https://huggingface.co/) and [Llama 2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 🦙🦙 model. With these tools, you can easily develop a web application that is user-friendly and allows for natural language questioning from a PDF document. This solution is both simple and effective, enabling users to extract valuable information from the document through semantic searching.

I referred to Andrew Ng's book, "[Machine Learning Yearning](https://info.deeplearning.ai/machine-learning-yearning-book)" to embed data in the local [Chroma](https://www.trychroma.com/) vector database. This allows us to perform similarity searches on user inquiries from the database. We can then use the Llama 2 model to summarize the results and provide feedback to the user.

Both the Embedding and LLM (Llama 2) models can be downloaded and run on your local machine. This allows for use in private environments without an internet connection. However, it is recommended to have a relatively powerful machine, ideally with a GPU, to achieve higher response performance when running Llama 2.

* [preprocess_chroma.ipynb](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/preprocess_chroma.ipynb) <-- Example of using Embedding Model from Open-Source Embedding Model ("sentence-transformers/all-MiniLM-L6-v2") in Hugging Face to embed the content from the document and save it into Chroma vector database.
* [consume_chroma.ipynb](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/consume_chroma.ipynb) <-- Example of using LangChain question-answering module to perform similarity search from the Chroma vector database and use the Llama 2 model to summarize the result.
* [consume_chroma.ipynb](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/consume_chroma_apple_silicon.ipynb) <-- Example of using LangChain question-answering module to perform similarity search from the Chroma vector database and use the Llama 2 model to summarize the result. Special version of Apple Silicon chip for GPU Acceleration (Tested work in MBA M2 2022).
* [app.py](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/app.py) <-- Example of using Streamlit, LangChain, and Chroma vector database to build an interactive chatbot to facilitate the semantic search over documents. It uses the Llama 2 model for result summarization and chat.


![alt text](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/git-images/git-image-1.png)


**Architecture**
![alt text](https://github.com/easonlai/chat_with_pdf_streamlit_llama2/blob/main/git-images/git-image-2.png)


**To run this Streamlit web app**
```
streamlit run app.py
```

**Enjoy!**

