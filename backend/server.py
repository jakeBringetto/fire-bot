from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model import user_query

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/")
def home():
    return {"message": "Hello from backend"}

@app.route("/model_query/<input_>")
def model_query(input_):
    response = user_query.query(str(input_))
    return jsonify({"message": response})

if __name__ == '__main__':
    app.run(debug=True)