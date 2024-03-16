from flask import Flask, render_template, request
from backend import get_api_keys, connect_to_pinecone, setup_vectorstore, generate_retrieval_answer

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the HTML page located at "templates/index.html"
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    # This function handles form submission and displays the generated question.
    question = request.form.get('question')
    OPENAI_API_KEY, PINECONE_API_KEY = get_api_keys()
    index = connect_to_pinecone(PINECONE_API_KEY)
    vectorstore = setup_vectorstore(index, OPENAI_API_KEY)
    answer = generate_retrieval_answer(question, vectorstore, OPENAI_API_KEY)
    return render_template('index.html', question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)