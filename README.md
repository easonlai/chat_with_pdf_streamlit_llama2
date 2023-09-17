# Semantic Search over Documents (Chat with PDF) with Llama 2 🦙 & Streamlit 🌠

In this repository, you will discover how [Streamlit](https://streamlit.io/), a Python framework for developing interactive data applications, can work seamlessly with the Open-Source Embedding Model ("[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)") in Hugging Face and [Llama 2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 🦙🦙 model. With these tools, you can easily develop a web application that is user-friendly and allows for natural language questioning from a PDF document. This solution is both simple and effective, enabling users to extract valuable information from the document through semantic searching.

I referred to Andrew Ng's book, "[Machine Learning Yearning](https://info.deeplearning.ai/machine-learning-yearning-book)" to embed data in the local [Chroma](https://www.trychroma.com/) vector database. This allows us to perform similarity searches on user inquiries from the database. We can then use the Llama 2 model to summarize the results and provide feedback to the user.



**To run this Streamlit web app**
```
streamlit run app.py
```

Enjoy!

