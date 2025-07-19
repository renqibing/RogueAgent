import re
import json
from collections import defaultdict

log_filename = "social.agent-2025-02-09_15-37-58.log"
with open(f'./log/{log_filename}', 'r') as file:
    log_data = file.read()

import os

# Sample log data (replace this with reading from a file or other source)
# log_data = '''<insert the log content here>'''

# Step 1: Split the log by "INFO"
log_lines = log_data.split("INFO")

# Step 2: Define a function to extract agent actions and write them to their respective files
def extract_and_save_actions(log_lines):
    # Create a dictionary to store agent actions
    agent_actions = {}

    # Step 3: Process each log line
    for line in log_lines:
        line = line.strip()  # Remove leading/trailing whitespace
        
        if not line:
            continue
        
        # Check if the line mentions an agent (e.g., "Agent 1", "Agent 2")
        agent_match = None
        for i in range(0, 1000):  # We assume up to 9 agents (can adjust for more agents)
            agent_match = f"Agent {i}"
            if agent_match in line:
                break

        if agent_match:
            # Ensure the agent's file exists
            if agent_match not in agent_actions:
                agent_actions[agent_match] = []

            # Add the current log line to the corresponding agent's actions
            agent_actions[agent_match].append(line)
    
    # Step 4: Save actions to individual files
    output_dir = 'agent_actions'
    os.makedirs(output_dir, exist_ok=True)

    for agent_id, actions in agent_actions.items():
        file_path = os.path.join(output_dir, f'{agent_id}_actions.txt')
        with open(file_path, 'w') as f:
            f.write("\nINFO".join(actions))  # Reassemble the action with "INFO" prefix as separator
            
    print(f"Actions saved to separate files in the '{os.path.join(os.getcwd(), 'agent_actions')}' directory.")

# Call the function with the log data
extract_and_save_actions(log_lines)
