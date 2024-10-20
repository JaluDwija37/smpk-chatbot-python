from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import os
import sys
import signal
import time

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
def run_train_and_bot():
    python_executable = sys.executable
    subprocess.run([python_executable, 'training.py'], check=True)
    subprocess.run([python_executable, 'main.py'], check=True)


@app.route('/trigger', methods=['POST'])
def trigger_bot():
    try:
        run_train_and_bot()  # Run train.py and main.py
        return jsonify({"message": "Bot triggered successfully!"}), 200
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