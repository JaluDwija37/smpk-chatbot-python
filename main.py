from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import sys
import signal
import time
import requests
import json

app = Flask(__name__)
CORS(app)

# Global variable to keep track of the main bot process
main_process = None

def stop_main():
    """Stop the main.py process."""
    global main_process
    if main_process:
        print("Sending shutdown request to main.py...")
        os.kill(main_process.pid, signal.SIGTERM)  # Send shutdown signal
        time.sleep(1)  # Wait for a second for the process to shut down

def send_status(learning_rate, batch_size, optimizer, testing_percentage, message="No message", status="200", **kwargs):
    """Send status update to monitoring endpoint."""
    try:
        response = requests.post(
            'http://127.0.0.1:8000/api/chatbot_status',
            json={
            "status": status,
            "message": message,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "testing_percentage": testing_percentage,
            "accuracy": kwargs.get("accuracy", 0),
            "precision": kwargs.get("precision", 0),
            "recall": kwargs.get("recall", 0)
            }
        )
        return response.ok
    except requests.RequestException:
        print("Failed to send status update")
        return False

@app.route('/trigger', methods=['POST'])
def trigger_bot():
    """Endpoint to trigger bot training and execution."""
    try:
        # Extract parameters from the request
        data = request.json
        learning_rate = data.get('learning_rate', '0.01')
        batch_size = data.get('batch_size', '32')
        optimizer = data.get('optimizer', 'adam')
        testing_percentage = data.get('testing_percentage', '0.2')

        # Run the processes asynchronously
        global main_process
        subprocess.run(
            [
                sys.executable, 
                'training-with-graph.py', 
                '--learning_rate', learning_rate, 
                '--batch_size', batch_size, 
                '--optimizer', optimizer, 
                '--testing_percentage', testing_percentage
            ], 
            check=True
        )
        main_process = subprocess.Popen([sys.executable, 'main.py'])
        try:
            with open('metrics.json', 'r') as metric_file:
                metrics = json.load(metric_file)
        except FileNotFoundError:
            metrics = {"accuracy": 0, "precision": 0, "recall": 0}  # Default values if file doesn't exist
        except json.JSONDecodeError:
            return jsonify({"message": "Error decoding metrics.json"}), 500

        send_status(
            message="Bot started successfully!", 
            status="200", 
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer,
            testing_percentage=testing_percentage,
            **metrics
        )
        return jsonify({"message": "Bot trained successfully!"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"message": f"Error in training: {str(e)}"}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Endpoint to shutdown the Flask app itself."""
    stop_main()  # Stop the main.py process
    return jsonify({"message": "Flask server shutting down..."}), 200

if __name__ == '__main__':
    print('Starting flask app...')
    app.run(host='0.0.0.0', port=8080)