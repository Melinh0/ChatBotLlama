from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
import httpcore  
from model import *
from config import *

app = Flask(__name__)
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/api/question', methods=['POST'])
def post_question():
    json_data = request.get_json(silent=True)
    if json_data is None or 'question' not in json_data or 'user_id' not in json_data:
        logging.error("Invalid JSON payload")
        return jsonify({"error": "Invalid JSON payload"}), 400

    question = json_data['question']
    user_id = json_data['user_id']
    logging.info("post question `%s` for user `%s`", question, user_id)

    try:
        resp = chat(question, user_id)
        data = {'answer': resp}
        return jsonify(data), 200
    except httpcore.ReadTimeout:
        logging.error("Request to external service timed out")
        return jsonify({"error": "Request timed out"}), 500
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    try:
        init_llm()
        index = init_index(Settings.embed_model)
        init_query_engine(index)
    except Exception as e:
        logging.error("Initialization error: %s", str(e))
        sys.exit(1)

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
