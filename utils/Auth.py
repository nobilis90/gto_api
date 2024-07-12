import os
import json
import secrets
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

load_dotenv()



# Define the path for the key and tokens file
#key_path = "auth/secret.key"
file_path = "auth/tokens.json"

# Generate and save the key (do this once and store it securely)
# def generate_and_save_key(key_path):
#     key = Fernet.generate_key()
#     with open(key_path, 'wb') as key_file:
#         key_file.write(key)

# Load the key from the file
# def load_key(key_path):
#     if not os.path.exists(key_path):
#         generate_and_save_key(key_path)
#     with open(key_path, 'rb') as key_file:
#         return key_file.read()

# Initialize Fernet with the loaded key
try:
    key = os.getenv("SECRET_KEY")
except Exception as err:
    logging.error("Environment variables is missing SECRET_KEY")
    raise SystemExit()

#key = load_key(key_path)
cipher_suite = Fernet(key)

def generate_token(length: int = 16) -> str:
    return secrets.token_urlsafe(length)

def encrypt_token(token: str, user_type: str) -> str:
    if user_type == "admin":
        token += "==admin"
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    try:
        decrypted_token = cipher_suite.decrypt(encrypted_token.encode()).decode()

        role = "user"
        if decrypted_token.endswith("==admin"):
            role = "admin"
            decrypted_token = decrypted_token[:-7]  # Remove the "==admin" suffix

        return {"decrypted_token" : decrypted_token, "role" : role}
    except InvalidToken as e:
        return {"decrypted_token" : "Error decrypting token", "role" : "None"}




# This function will load encrypted tokens from the file
def load_tokens(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as file:
        return json.load(file)

# This function will save encrypted tokens to the file
def save_tokens(file_path: str, tokens: dict) -> None:
    with open(file_path, 'w') as file:
        json.dump(tokens, file)

# This function will add a new token for a user
def add_user_token( user_email: str, user_type: str) -> str:
   
    new_token = generate_token()
    encrypted_token = encrypt_token(new_token, user_type)
    tokens = load_tokens(file_path)
    
    if user_email not in tokens:
        tokens[user_email] = {}
        
    # Overwrite the existing role token
    tokens[user_email][user_type] = new_token

    save_tokens(file_path, tokens)
    return encrypted_token



# # Example usage
# user_email = "nicolas.vizaccaro@stellantis.com"
# user_type = "admin"  # or "user"
# try:
#     new_token = add_user_token(user_email, user_type)
#     print(f"New token for {user_email} ({user_type}): {new_token}")
# except ValueError as e:
#     print(e)

# # Load and decrypt tokens
# tokens = load_tokens(file_path)
# for user_email, roles in tokens.items():
#     for user_type, encrypted_token in roles.items():
#         print(f"User Email: {user_email}, Type: {user_type}, Encrypted Token: {encrypted_token}")
#         decrypted_token = decrypt_token(new_token)
#         print(decrypted_token)
#         #if decrypted_token:
#         #    print(f"User Email: {user_email}, Type: {user_type}, Decrypted Token: {decrypted_token}")
