from fastapi import FastAPI
from llama_index import GPTVectorStoreIndex, download_loader, LLMPredictor
from llama_index.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings


llm = AzureOpenAI(
    engine="gpt-35-turbo",
    model="gpt-35-turbo",
    temperature=0.7,
    api_base="https://shiyanjia-ai-e2.openai.azure.com/",
    api_key="adb0b09568ac4b7a8f855f3287374170",
    api_type="azure",
    api_version="2023-03-15-preview",
    deployment_name="gpt-35-turbo"
)


llm_predictor = LLMPredictor(llm=llm)

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "Hello World"}

@app.get("/chat")
async def chat(question: str) -> dict:
    llm.complete(question)
    BilibiliTranscriptReader= download_loader("BilibiliTranscriptReader")
    loader = BilibiliTranscriptReader()
    documents = loader.load_data(video_urls=['https://www.bilibili.com/video/BV1yx411L73B/'])
    index = GPTVectorStoreIndex.from_documents(documents)
    index.query('Where did the author go to school?')
