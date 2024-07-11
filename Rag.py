from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Load documents from web
documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["https://www.formula1.com/en/results.html/2024/drivers.html"]
)

# Set up Ollama embedding and LLM
ollama_embedding = OllamaEmbedding(
    model_name="qwen2",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
Settings.embed_model = ollama_embedding
Settings.llm = Ollama(model="qwen2")

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine()

# Function to handle user queries
def process_query(query):
    response = query_engine.query(query)
    return response

# Example usage
query = "Who is the best Formula 1 driver in 2024?"
response = process_query(query)
print(response)