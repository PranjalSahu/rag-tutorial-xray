import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

topk = 50
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Below is a list of radiology reports that are present in the database.
You are a radiology expert and you have to make a list of reports from the 
following reports:

{context}

---

Given the above context, return the list of reports that satisfies the following question: {question}

Please return the reports in the following format:
Report: the report text
Metadata: the metadata that you found in the report
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)
    exit(0)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=topk)
    if False:
        print("----------------------------")
        counter = 0
        for k in results:
            print(counter, k[0])
            counter =  counter + 1
            print("----------------------------")
    
    context_text = "\n\n---\n\n".join(["Report Text: " + doc.page_content + " \nMetadata Source: " + doc.metadata['source'] for doc, _score in results])
    #print(context_text)

    #print(PROMPT_TEMPLATE)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    #print(prompt)

    model = Ollama(model="llama3", temperature=0.001)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    #print(sour)
    #formatted_response = f"Response: {response_text}\nSources: {sources}"
    formatted_response = f"Response: {response_text}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
