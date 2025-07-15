import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import time
from audio_to_text import transcribe_video_to_text
from langchain_community.vectorstores import FAISS
from text_to_chunks import transcribed_text_to_chunks
from langchain.vectorstores import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pytube import YouTube
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.llms import AI21
import pytube
import yt_dlp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


from langchain_core.language_models.llms import LLM

from huggingface_hub import InferenceClient

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

class HFChatLLM(LLM):
    model: str
    token: str

    def _call(self, prompt: str, stop=None):
        client = InferenceClient(api_key=self.token)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

class CustomHFClient:
    def __init__(self, model_name, token):
        self.client = InferenceClient(model=model_name, token=token)

    def __call__(self, prompt, **kwargs):
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 100)
        )
        return response.choices[0].message["content"]


# Function to generate a welcoming message in HTML format
def generate_welcoming_message():
    markdown = """
    <style>
    ...
    </style>
    <div class="container">
        <div>
            <h1 class="title">👋 Welcome to YouTube ChatBot!</h1>
            <p class="description">Please enter a YouTube video link above and start chatting</p>
        </div>
    </div>
    """
    return markdown

# Function to get the vectorstore from chunks using HuggingFaceEmbeddings and FAISS
def get_vectorstore(chunks):
    try:
        embed = HuggingFaceEmbeddings(encode_kwargs={'normalize_embeddings': True})
        vectorstore = FAISS.from_documents(chunks, embed)
        return vectorstore
    except Exception as e:
        st.sidebar.error("please enter a valid link and let's chat again ", icon="🚨")
        print(e)

# Function to create an advanced conversation chain using EnsembleRetriever
def get_conversation_chain_advanced(vectorstore):
    loader = DirectoryLoader(r"D:\M.Tech\Projects\YouTube-AI-Assistant\Youtube",
                             glob='*.txt',
                             loader_cls=TextLoader)

    documents = loader.load()
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True
    )
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5])
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True,
        get_chat_history=lambda h: h,
        verbose=False
    )
    return chain

def get_video_chara(url):
    try:
        ydl_opts = {'quiet': True, 'skip_download': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title')
            video_duration = info.get('duration')
            video_thumbnail = info.get('thumbnail')
            return video_title, video_duration, video_thumbnail
    except Exception as e:
        st.sidebar.error("❌ Failed to fetch metadata. Video may be age-restricted or broken link.", icon="🚨")
        st.stop()

# Cache for storing summaries
summaries_cache = {}

# Function to get a summary for a video based on its key
def get_summary(video_key, documents):
    if video_key not in summaries_cache:
        prompt_template = """Write a concise summary of the following "{context}"
                             CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Correct usage with 'context'
        stuff_chain = create_stuff_documents_chain(llm, prompt)

        # Use invoke instead of run
        summary = stuff_chain.invoke({"context": documents})
        summaries_cache[video_key] = summary

    return summaries_cache[video_key]

# Function to find and highlight a sentence in a paragraph
def find_and_highlight_sentence(paragraph, target_sentence, transcribed_text):
    if target_sentence in transcribed_text:
        print("✅ Match found — length:", len(target_sentence))
        highlighted_sentence = f"<span style=' color: red;'>{target_sentence}</span>"
        return paragraph.replace(target_sentence, highlighted_sentence)
    else:
        print("❌ No match")
    return paragraph

# Setting up environment variables
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN 
llm = HFChatLLM(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=HF_TOKEN
)

# Initializing Streamlit session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'empty_space' not in st.session_state:
    st.session_state.empty_space = st.empty()
    welcoming_message = generate_welcoming_message()
    st.session_state.empty_space.markdown(welcoming_message, unsafe_allow_html=True)
if 'num' not in st.session_state:
    st.session_state.num = None

# Sidebar layout for the YouChat App
st.sidebar.markdown(
    f"<h1 style='color: #EC5331; font-size: 36px; font-weight: bold;'>YouChat App</h1>",
    unsafe_allow_html=True
)
st.sidebar.title('▶️ Provide a YouTube Video Link')
url = st.sidebar.text_input("Paste the YouTube Video Link here", value="", help="E.g., https://www.youtube.com/watch?v=your_video_id")
if url != "":
    st.session_state.num = True
col1, col2 = st.sidebar.columns(2)
with col1:
    ask_button = st.button("Analyze Video 🚀")
with col2:
    reset_button = st.button("Reset Form 🔄")

# Resetting form and session state variables if reset button is clicked
if reset_button:
    directory = r'D:\M.Tech\Projects\YouTube-AI-Assistant\Youtube'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    url = ""
    st.session_state.messages = []
    ask_button = True
    if st.session_state.num:
        st.sidebar.info("please enter a new video link and let's chat again", icon="ℹ️")
    st.session_state.conversation = None
    st.session_state.documents = None

if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = None

# Analyzing the video and processing user input
if url != "":
    if ask_button:
        with st.sidebar:
            st.session_state.messages = []
            st.session_state.conversation = None
            with st.spinner("Analyzing the Video... 🕵️‍♂️"):
                ask_button = False
                st.session_state.transcribed_text = transcribe_video_to_text(url)

                print("\n📝 Transcript sample (first 500 chars):\n", st.session_state.transcribed_text[:500])
                print("📏 Transcript length:", len(st.session_state.transcribed_text))
                st.session_state.documents, st.session_state.chunks = transcribed_text_to_chunks(r'D:\M.Tech\Projects\YouTube-AI-Assistant\Youtube')
                print(st.session_state.chunks)
                st.session_state.vectorstore = get_vectorstore(st.session_state.chunks)
                print(st.session_state.vectorstore)
                if st.session_state.vectorstore is not None:
                    st.session_state.conversation = get_conversation_chain_advanced(st.session_state.vectorstore)

# Displaying video details and summary
if url != "":
    with st.sidebar:
        video_title, video_duration, video_thumbnail = get_video_chara(url)
        if video_title is not None and video_duration is not None and video_thumbnail is not None and st.session_state.documents is not None:
            summary = get_summary(url, st.session_state.documents)
            video_id = id = url.split("=")[-1]
            st.markdown("<h1 style='color: #EC5331;'>📽️ Video Details</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 18px;'><strong><span style='color:#EC5331;'>Summary:<br></span></strong> {summary}</p>",
                        unsafe_allow_html=True)
            st.markdown(
                f'<iframe width="490" height="315" src="https://www.youtube.com/embed/{video_id}?start={0}&autoplay=1" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True)
            transcript = st.markdown(
                f"<div style='height: 400px; overflow-y: scroll;'>{st.session_state.transcribed_text}</div>",
                unsafe_allow_html=True)

# Displaying chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'], unsafe_allow_html=True)

# Handling user input and generating responses
if st.session_state.conversation is not None:
    prompt = st.chat_input('Message to ChatBot...')
    if prompt:
        with st.chat_message('user'):
            st.markdown(prompt)

        st.session_state.messages.append({'role': 'user', 'content': prompt})
        response = f'Echo {prompt}'
        response = st.session_state.conversation(prompt)
        with st.spinner("Thinking...Please wait..."):
            time.sleep(1)
        with st.chat_message('assistant'):
            answer = response.get("answer")
            doc = response['source_documents']
            if doc:
                source = str(doc[0]).split("\\")[-1].replace(".txt'}", "")
                answer_final = f"{answer} \n\n Source: {video_title} Video"
                transcribed_text_highlited = st.session_state.transcribed_text
                for i in range(len(doc)):
                    print("\n📄 Retrieved chunk content:\n", doc[i].page_content)
                    print(f"Match found — length: {len(doc[i].page_content)}")

                    if len(doc[i].page_content) < 200:
                        transcribed_text_highlited = find_and_highlight_sentence(transcribed_text_highlited, doc[i].page_content, st.session_state.transcribed_text)

                transcript.markdown(
                    f"<div style='height: 400px; overflow-y: scroll;'>{transcribed_text_highlited}</div>",
                    unsafe_allow_html=True)
                print("🟠 Highlighted transcript length:", len(transcribed_text_highlited))

            else:
                source = 'your AI assistant'
                answer_final = f"{answer} \n\n Source: {source}"
            segments = answer_final.split("\n\n")
            segments[-1] = f"<span style='color: #EC5331;'>{segments[-1]}</span>"
            answer_final = f"{segments[0]} \n\n {segments[-1]}"
            st.markdown(answer_final, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': answer_final})