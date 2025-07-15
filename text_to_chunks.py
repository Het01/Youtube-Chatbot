from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def transcribed_text_to_chunks(url):
 loader = DirectoryLoader(url,
                             glob='*.txt',
                             loader_cls=TextLoader)

 documents = loader.load()
 text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=60,
 )
 chunks=text_splitter.split_documents(documents)
 for i, chunk in enumerate(chunks):
     print(f"📦 Chunk {i+1} - Length: {len(chunk.page_content)}")
     print(f"Content preview: {chunk.page_content[:100]}...\n")
 return documents,chunks
