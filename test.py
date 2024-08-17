from langchain.embeddings import HuggingFaceInstructEmbeddings

# Initialize the embeddings with the model
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

# Use the embeddings (example)
texts = ["Hello, world!", "How are you?"]
embeddings_output = embeddings.embed(texts)
print(embeddings_output)