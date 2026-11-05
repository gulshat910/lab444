import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")   # чтобы VPN не мешал Ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama
from src.hybrid_retriever import retriever

llm = ChatOllama(model=os.getenv("LLM_MODEL", "llama3.1:latest"),
                 base_url="http://127.0.0.1:11434",
                 temperature=0.2, num_ctx=4096)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты отвечаешь на вопросы на основе переданного контекста.\n"
     "Используй только информацию из контекста. Если точного ответа или цифры "
     "в контексте нет, так и скажи, ничего не додумывай.\n"
     "Отвечай на русском языке, точно и по делу. Ссылайся на источники в виде [номер].\n\n"
     "Контекст:\n{context}"),
    ("human", "Вопрос: {question}")])


def format_documents(documents):
    return "\n\n".join(f"[{i}] Источник: {d.metadata['title']}\n{d.page_content}"
                       for i, d in enumerate(documents, 1))


generate = (
    RunnableLambda(lambda x: {"context": format_documents(x["docs"]),
                              "question": x["question"]})
    | prompt | llm | StrOutputParser())

# вопрос -> {docs, question} -> добавляем answer
rag_chain = (RunnableParallel(docs=retriever, question=RunnablePassthrough())
             .assign(answer=generate))


def generate_answer(question):
    result = rag_chain.invoke(question)
    return result["answer"], result["docs"]


if __name__ == "__main__":
    ans, docs = generate_answer(input("Введите вопрос: "))
    print(ans)
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata['title']} ({d.metadata['url']})")
