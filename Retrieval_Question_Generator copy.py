import os
from flask import Flask, render_template, request, json
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain



app = Flask(__name__)
print(app.url_map)

def get_api_keys():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("API keys for OpenAI and Pinecone must be set as environment variables.")
    return OPENAI_API_KEY, PINECONE_API_KEY

def connect_to_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    index_name = 'lecture-data'
    index = pc.Index(index_name)
    print("Connected to Pinecone index:", index_name)
    return index

def setup_vectorstore(index, api_key):
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    text_field = "text"
    vectorstore = LC_Pinecone(index, embed.embed_query, text_field)
    return vectorstore

def generate_retrieval_answer(question, vectorstore, api_key):
    llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-4-turbo-preview', temperature=0.4)
    template = """
    Du bist ein Fragen generierender Bot für Studenten des Studiengangs Information Systems.
    Deine Aufgabe ist es Single Choice Questions zu erstellen! 

    "Welche Aussage(n) zum Thema ______ ist/sind korrekt?

    a)
    b)
    c)
    d)
    
    Quelle:"


    
    Von den Aussagen ist immer eine Korrekt und die restlichen falsch. Gebe die richtige Antwort NICHT mit an.
    Sowohl die Frage als auch die eine richtige Aussage sollten AUSSCHLIEßLICH auf Basis des bereitgestellten Kontext erstellt werden.
    Bei den falschen Aussagen kannst du kreativ sein und dir inkorrekte aussagen einfallen lassen, die zwar richtig klingen, jedoch falsch sind.
    
    Das Thema der Frage solltest du dem Prompt herauslesen.
    Gebe am Ende den verwendeten Kontext als Quelle oder Quellen - sofern es mehrere sind - mit an. Zitiere ausschließlich Quellen aus dem gegebenen Kontext.
    Sollte der gegebene kontext nichts mit dem Prompt zu tun haben, erstelle keine Frage! Stattdessen frage danach, ein Thema aus dem EWI-Kurs zu wählen.
    
    Kontext:
    {summaries}
    Prompt:
    {question}
    """

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={
            "prompt": PromptTemplate(template=template, input_variables=["summaries", "question"])
        }
    )

    result = chain(question)
    # Simplifying and formatting the output here for web display
    extracted_answer = {
            result.get('answer')
    }
    return extracted_answer


@app.route('/')
def index():
    # Renders the HTML page located at "templates/index.html"
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    # This function handles form submission and displays the generated question.
    question = request.form.get('question')
    answer = generate_retrieval_answer(question, vectorstore, OPENAI_API_KEY)
    return render_template('index.html', question=question, answer=answer)

if __name__ == "__main__":
    OPENAI_API_KEY, PINECONE_API_KEY = get_api_keys()
    index = connect_to_pinecone(PINECONE_API_KEY)
    vectorstore = setup_vectorstore(index, OPENAI_API_KEY)
    app.run(debug=True)
