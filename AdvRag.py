from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_models import ChatOllama

# Load documents (you can replace this with your own data loading method)
docs = [
    Document(
        page_content="Max Verstappen dominates the 2024 Formula 1 season with multiple wins",
        metadata={"year": 2024, "team": "Red Bull Racing", "wins": 10},
    ),
    Document(
        page_content="Lewis Hamilton shows strong performance in his new team",
        metadata={"year": 2024, "team": "Ferrari", "podiums": 8},
    ),
    # Add more documents as needed
]

# Create vector store
vectorstore = Chroma.from_documents(docs, OllamaEmbeddings(model="qwen2"))

# Define metadata field info
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="The year of the Formula 1 season",
        type="integer",
    ),
    AttributeInfo(
        name="team",
        description="The Formula 1 team name",
        type="string",
    ),
    AttributeInfo(
        name="wins",
        description="Number of race wins",
        type="integer",
    ),
    AttributeInfo(
        name="podiums",
        description="Number of podium finishes",
        type="integer",
    ),
]

# Set up LLM
llm = ChatOllama(model="qwen2", temperature=0)

# Create self-query retriever
document_content_description = "Information about Formula 1 drivers and their performance"
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

# Function to process queries
def process_query(query):
    return retriever.invoke(query)

# Example usage
query = "Who has the most wins in the 2024 season?"
result = process_query(query)
print(result)