import os
import re
import csv
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

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

def load_lectures_from_csv(file_path):
    lectures = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            lecture = row['Lecture']
            topic = row['Topic']
            if lecture in lectures:
                lectures[lecture].append(topic)
            else:
                lectures[lecture] = [topic]
    return lectures

def generate_retrieval_answer(question, vectorstore, api_key):
    llm = ChatOpenAI(
    openai_api_key=api_key,
    #model_name='gpt-3.5-turbo',
    model_name='gpt-4-turbo-preview',
    temperature=0.5
    #response_format = {type: "json_object"}
    )
    template = """
    Du bist ein Fragen generierender Bot für Studenten des Studiengangs Information Systems.
    Deine Aufgabe ist es Multiple Choice Questions zu erstellen!
    Erstelle die Fragen im JSON format mit den Schlüsseln: 
    
    Frage, 
    Aussagen (soll 10 Aussagen a bis j enthalten), 
    Richtige_Antworten (Enthält eine Liste der richtigen Antworten),
    Quellen (Enthält eine liste der verwendeten Quellen inkl. Seite)

    Es soll eine Mischung aus richtigen und falschen Aussagen erstellt werden. Mindestens eine und maximal 9 Aussagen sollen korrekt sein. Gebe die richtigen Aussagen unter dem Punkt "Richtige_Antworten" mit an. Die richtigen und falschen Aussagen sollen zufällig positioniert sein.
    Die richtigen Aussagen sollten AUSSCHLIEßLICH auf Basis des bereitgestellten Kontext erstellt werden.
    Bei den falschen Aussagen kannst du kreativ sein und dir inkorrekte aussagen einfallen lassen, die zwar richtig klingen, jedoch falsch sind. Trenne die Aussagen a) bis j) IMMER mit einer leeren zeile.
    
    Das Thema der Frage solltest du dem Prompt herauslesen.
    Verwende NUR Quellen inkl. Seite die innerhalb des gegebenen Kontext unter Source agegeben sind!
    
    Kontext:
    {summaries}
    Prompt:
    {question}
    """

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), #retrieve the document junks and define how many end in the prompt
        chain_type_kwargs={
            "prompt": PromptTemplate(template=template, input_variables=["summaries", "question"])
        }
    )

    result = chain(question)
    raw_text = result.get('answer')

    # Define a regex pattern to match the desired format
    pattern = r'^\s*```json\s+(.*?)\s*```\s*$'
    match = re.match(pattern, raw_text, re.DOTALL)

    if match:
        # Extract the text without the markers
        cleaned_text = match.group(1)
    else:
        # If the format does not match, leave the text as is
        cleaned_text = raw_text


    # Use `cleaned_text` for further processing
    extracted_answer = cleaned_text
    return extracted_answer
