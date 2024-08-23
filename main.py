from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.bedrock import BedrockEmbeddings

DATA_PATH = 'data'

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=80,
        length_function = len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embeddings_function():
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-1"
    )
    return embeddings

documents = load_documents()

chunks = split_documents(documents)

print(chunks[1])

