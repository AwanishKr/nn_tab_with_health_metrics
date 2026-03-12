import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')

def load_config(path=CONFIG_PATH):
    """Load configuration from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
