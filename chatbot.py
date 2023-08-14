import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="GS_CourseOverview.csv")
documents = loader.load()

#print(len(documents))

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(documents, embeddings)
print(db)

# 2. Function for similarity search


def retrieve_info(query, db):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

question="How does your research play into your course design?"

results = retrieve_info(question, db)

print(results)

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are UCLA Business school professor who teaches a class about Real Estate Technology. 
You are speaking to a journalist from UCLA's student newsletter.
I will share a question from the journalist. And you will give me the answer that
would best answer that question based on past answers and you will follow ALL of the rules below:

1/ Responses should be similar or even identical to past answers in terms of length, tone of voice, 
logical arguments and other details
2/ If the answers are irrelevant, then try to mimic the style of past answers as well as reflect
relevant and accurate information about real estate technology.  
3/ Keep the answers punchy and professional

Below is a message I received from the prospect:
{question}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{answer}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(question,db):
    answer = retrieve_info(question, db)
    response = chain.run(question=question, answer=answer)
    return response

#question = "How does your research play into your course design?"
#response = generate_response(question, db)
#print(response)


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Professor response generator", page_icon=":bird:")

    st.header("Professor response generator :bird:")
    message = st.text_area("What would like to ask the professor?")

    if message:
        st.write("Generating the Professor's message...")

        result = generate_response(message,db)

        st.info(result)


if __name__ == '__main__':
    main()