from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import sys
import signal
import time
import requests

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

def send_status(message="No message", status="200"):
    """Send status update to monitoring endpoint."""
    try:
        response = requests.post(
            'http://127.0.0.1:8000/api/chatbot_status',
            json={"status": status, "message": message}
        )
        return response.ok
    except requests.RequestException:
        print("Failed to send status update")
        return False

@app.route('/trigger', methods=['POST'])
def trigger_bot():
    """Endpoint to trigger bot training and execution."""
    try:
        # Run the processes asynchronously
        global main_process
        subprocess.run([sys.executable, 'training.py'], check=True)
        main_process = subprocess.Popen([sys.executable, 'main.py'])
        send_status("Bot started successfully!")
        return jsonify({"message": "Bot Trained successfully!"}), 200
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