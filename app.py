import streamlit as st
import yt_dlp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import DeepLake
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-001", api_key=GEMINI_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)

def download_mp4_from_youtube(url, job_id):
    file_temp = f'./{job_id}.mp4'
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': file_temp,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        title = result.get('title', "")
        author = result.get('uploader', "")
        thumbnail_url = result.get('thumbnail', "")

    return (file_temp, title, author, thumbnail_url)


def transcription(video_details):
    client = Groq(api_key=GROQ_API_KEY)
    filename, title, author, _ = video_details

    with open(filename, "rb") as file:
        result = client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="distil-whisper-large-v3-en",
            response_format="json",
            temperature=0.0
        )

    with open('text.txt', 'w') as file:
        file.write(result.text)

    return 'text.txt'

def split_text(filename):
    with open(filename) as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    texts = text_splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts[:4]]
    return docs

def create_embeddings(docs):
    dataset_path = "vector_database"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
    db.add_documents(docs)

    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 4
    return retriever


def retrieve_response(retriever, question):
    prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Summarized answer in bullet points:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs={"prompt": PROMPT})

    response = qa.run(question)
    return response

def main():
    st.title("YouTube Video Summarizer")

    url = st.text_input("Enter a YouTube URL")

    if st.button("Process Video"):
        if url:
            st.write("Downloading video...")
            video_info = download_mp4_from_youtube(url, job_id=1)

            _, title, author, thumbnail_url = video_info
            st.image(thumbnail_url, caption=f"{title} by {author}")

            text_file = transcription(video_info)
            docs = split_text(text_file)
            retriever = create_embeddings(docs)

            st.write("### Summarized Response:")
            summary_question = "Summarize the main points of the video"
            summary_response = retrieve_response(retriever, summary_question)
            st.write(summary_response)

            user_question = st.text_input("Ask a question about the video")
            if user_question:
                answer = retrieve_response(retriever, user_question)
                st.write(f"### Answer to your question: {answer}")
        else:
            st.warning("Please enter a YouTube URL.")

if __name__ == "__main__":
    main()