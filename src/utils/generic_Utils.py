import uuid
import yaml
import os, json
from dotenv import load_dotenv

def generate_formatted_uuid():
  """
  Generates a UUID in the format "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".

  The uuid.uuid4() method naturally produces UUIDs in this standard string format.
  """
  # Generate a version 4 (random) UUID
  # The __str__ method of the UUID object automatically formats it
  # as "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
  new_uuid = uuid.uuid4()

  # Convert the UUID object to its string representation
  formatted_uuid_string = str(new_uuid)

  return formatted_uuid_string


def load_config(env='development'):
    base_config_path = 'src/configs/config.yaml'
    env_config_path = f'src/configs/config.{env}.yaml'

    config = {}

    # Load base config
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # Load environment-specific config (overrides base)
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    # Substitute environment variables
    config = _substitute_env_vars(config)

    return config

def _substitute_env_vars(config):
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        var_name = config[2:-1]
        return os.getenv(var_name, config)
    return config

# Load .env from external secrets folder
# Priority: ENV_FILE_PATH environment variable > External secrets folder > Local .env
from pathlib import Path

# Try both 'env' and '.env' in secrets folder
external_paths = [
    Path(r'C:\Users\Pucci\Desktop\secrets\env'),
    Path(r'C:\Users\Pucci\Desktop\secrets\.env'),
]

env_path = os.getenv('ENV_FILE_PATH')
if not env_path:
    for path in external_paths:
        if path.exists():
            env_path = str(path)
            break
    if not env_path:
        env_path = '.env'

# Load environment variables
# Note: dotenv expects files ending in .env, so we manually parse if needed
if Path(env_path).exists():
    success = load_dotenv(dotenv_path=env_path, override=True)

    # If load_dotenv failed (likely because file doesn't end in .env), manually load
    if not success or not os.getenv('MONGODB_URI'):
        print(f"Manually parsing environment file: {env_path}")
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"').strip("'")
                        os.environ[key.strip()] = value
            print(f"[OK] Environment loaded manually from: {env_path}")
        except Exception as e:
            print(f"[ERROR] Error loading environment: {e}")
    else:
        print(f"[OK] Environment loaded via dotenv from: {env_path}")
else:
    print(f"[ERROR] Environment file not found: {env_path}")

# Verify critical variables loaded
if os.getenv('MONGODB_URI'):
    print("[OK] MONGODB_URI loaded")
if os.getenv('GEMINI_API_KEY'):
    print("[OK] GEMINI_API_KEY loaded")

environment = os.getenv("environment", "development")

CONFIG = load_config(os.getenv('APP_ENV', environment))

def add_batchId_fileId(jsonOutput, batchId, fileId):
    """Read a JSON file, add batchId and fileId to each item (if list) or to the dict, update the file, and return the modified JSON object."""

    data = json.loads(jsonOutput)
    if isinstance(data, list):
        for item in data:
            item["batchId"] = batchId
            item["fileId"] = fileId
    return data


def get_initial_input_from_state(previousMessages, initialInput):
    """Extracts the initial input from the given state dictionary."""

    history_messages = previousMessages
    current_query = initialInput
    print(f"Decision Taking Node messages: {history_messages}")

    # 2. Format the history part
    history_str = '\n'.join([f"{role}: {msg}" for role, msg in history_messages])

    # 3. Create the new, formatted input string
    # We add the "Previous Messages Context" only if there is history
    if history_str:
        modified_input = f"""Previous Messages Context : {history_str}
                            User Query : {current_query}"""
    else:
        # Handle the first message (no history)
        modified_input = f"User Query : {current_query}"

    return modified_input

def get_scope_description (scope: str) -> str:
    """Returns a description based on the provided scope."""
    scope = scope.lower()
    if scope == 'batchlist':
        return "Batch overview and management expert specializing in batch processing, tracking, and optimization."
    elif scope == 'batchdetails':
        return "Batch details specialist with expertise in analyzing the extracted data."
    elif scope == 'contractanalytics':
        return "Contract analytics expert skilled in reviewing and optimizing contract performance."
    elif scope == 'complianceanalytics':
        return "Compliance analytics specialist focused on ensuring adherence to regulations and standards."
    elif scope == 'contractinsights':
        return "Contract insights specialist dedicated to uncovering valuable information from contracts."
    elif scope == 'reconciliation':
        return "Reconciliation expert focused on ensuring data accuracy and consistency across systems."
    elif scope == 'purchasepriceanalytics':
        return "Purchase price analytics specialist dedicated to optimizing procurement costs and supplier negotiations."
    elif scope == 'workingcapitalanalytics':
        return "Working capital analytics expert focused on improving liquidity and operational efficiency."
    else:
        return "You are a general expert ready to assist with a wide range of topics."
