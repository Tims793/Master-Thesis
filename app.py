from flask import Flask, render_template, request, session, jsonify
from flask_executor import Executor
import redis
import csv
import os
import json
import random
import string
from openai import OpenAI
client = OpenAI()
from backend import get_api_keys, connect_to_pinecone, setup_vectorstore, generate_retrieval_answer, load_lectures_from_csv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Connect to Redis
#redis_url = "redis://localhost:6379"  # Adjust as necessary
redis_url = os.getenv('REDIS_CLI')
redis_client = redis.Redis.from_url(redis_url)

app = Flask(__name__)

# create a session key
characters = string.ascii_letters + string.digits
app.secret_key = ''.join(random.choice(characters) for i in range(8))
# print("der erstellte key ist:" ,(app.secret_key))

executor = Executor(app)

lectures = load_lectures_from_csv('Lecture_topic_list.csv')

results = []

def prepare_next_response():
    # print("entered prepare_next_response")
    chosen_lectures = session.get('chosen_lectures', [])
    # print(chosen_lectures)
    chosen_lecture = random.choice(chosen_lectures)
    session['new_lecture'] = chosen_lecture
    topics = lectures.get(chosen_lecture, [])
    # print(topics)
    question = random.choice(topics)
    session['new_topic'] = question
    # print(question)
    OPENAI_API_KEY, PINECONE_API_KEY = get_api_keys()
    index = connect_to_pinecone(PINECONE_API_KEY)
    vectorstore = setup_vectorstore(index, OPENAI_API_KEY)
    text = generate_retrieval_answer(question, vectorstore, OPENAI_API_KEY)
    redis_client.set("button_enabled", "true")
    print("Button enabled set in redis")
    return(text)

@app.route('/')
def index():
    executor.futures.pop('prepared_response')  # Clearing executor's future if used
    return render_template('index.html', lectures=lectures.keys())

@app.route('/generate_question', methods=['POST'])
def generate_question():
    redis_client.set("button_enabled", "false")
    session['session_id'] = app.secret_key
    # Überprüfen, ob eine Antwort im Hintergrund vorbereitet wurde
    if executor.futures.done('prepared_response'):
        # set new topics and lectures to the current one (needed for the response generation)
        session['lecture'] = session['new_lecture']
        session['topic'] = session['new_topic']
        future = executor.futures.pop('prepared_response')
        parsed_text = json.loads(future.result())
        executor.submit_stored('prepared_response', prepare_next_response)  # Start preparing the next response again
        return render_template('index.html', answer=parsed_text)
    else:
        print("No prepared response ready.")

    chosen_lectures = request.form.getlist('lecture[]')
    print(chosen_lectures)
    session['chosen_lectures'] = chosen_lectures
    chosen_lecture = random.choice(chosen_lectures)
    session['lecture'] = chosen_lecture
    if chosen_lecture:
        topics = lectures.get(chosen_lecture, [])
        session['topics'] = topics
        session['chosen_lecture'] = chosen_lecture
    else:
        topics = session.get('topics', [])
        chosen_lecture = session.get('chosen_lecture', None)

    if not topics:
        return "No topics available for the selected lecture.", 404

    question = random.choice(topics)
    session['topic'] = question
    OPENAI_API_KEY, PINECONE_API_KEY = get_api_keys()
    index = connect_to_pinecone(PINECONE_API_KEY)
    vectorstore = setup_vectorstore(index, OPENAI_API_KEY)
    text = generate_retrieval_answer(question, vectorstore, OPENAI_API_KEY)
    parsed_text = json.loads(text)

    executor.submit_stored('prepared_response', prepare_next_response)  # Beginnen Sie sofort mit der Vorbereitung einer weiteren Antwort

    return render_template('index.html', answer=parsed_text)

@app.route('/check-button-status')
def check_button_status():
    button_enabled = redis_client.get("button_enabled")
    # print(button_enabled)
    return jsonify({"button_enabled": button_enabled.decode("utf-8") == "true"})

@app.route('/submit_result', methods=['POST'])
def submit_result():
    data = request.get_json()
    # print("Received and stored data:", data)
    # Initialize the score
    score = 0

    # Define the possible labels
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

    # Calculate the score
    for label in labels:
        if label in data['student_answer'] and label in data['correct_answer']:
                    score += 1
        elif label not in data['student_answer'] and label not in data['correct_answer']:
                    score += 1
        else:
                    score -= 1

    # Ensure score is not negative
    if score < 0:
        score = 0

    # Calculate percentage score
    max_score = len(labels)  # Maximum possible score
    percentage_score = (score / max_score) * 100

    result = {
        'lecture': session.get('lecture'),
        'topic': session.get('topic'),
        'student_answers': data['student_answer_texts'],
        'correct_answers': data['correct_answer_texts'],
        'percentage_score': percentage_score
    }
    #for value in result.values():
    #    print(value)

    results.append(result)
    return jsonify({'message': 'Result recorded'}), 200

@app.route('/get_feedback', methods=['POST'])
def get_feedback():
    feedback_text = generate_prompt()
    feedback = ask_gpt(feedback_text)
    return render_template('feedback.html', feedback=feedback)

def generate_prompt():
    feedback_text = "Du bist ein hilfreicher Tutor für Studenten des Kurses Einführung in die Wirtschaftsinformatik.\n"
    feedback_text += "Anbei erhälst du die Reslutate eines tests den ein Student online gemacht hat\n"
    feedback_text += "Vorlesung, Thema, Ausgewählte Aussagen, Richtige Aussagen, Bewertung\n"
    for result in results:
        feedback_text += f"{result['lecture']}, {result['topic']}, {result['student_answers']}, {result['correct_answers']}, {result['percentage_score']}\n"

    feedback_text += "\nGeneriere ein personalisiertes Feedback zu dem Test auf Basis der gegebenen Resultate. Halte dich kurz und fokussiere dich auf die Bereiche, in denen noch Verbesserungspotenzial besteht."
    return feedback_text

def ask_gpt(prompt):
    OPENAI_API_KEY = get_api_keys()

    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=500
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)