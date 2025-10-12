from cryptography.fernet import Fernet
import os

def generate_key(path: str = ".local/keys/key.key"):
    """
    Genera una nueva clave Fernet y la guarda en el archivo especificado.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = Fernet.generate_key()
    with open(path, "wb") as key_file:
        key_file.write(key)
    print(f"Nueva key generada en {path}")
    return key


def load_key(path: str = ".local/keys/key.key") -> bytes:
    """
    Carga la clave Fernet desde el archivo especificado.
    """
    with open(path, "rb") as key_file:
        return key_file.read()


def encrypt_file(input_path: str, output_path: str, key_path: str = ".local/keys/key.key"):
    """
    Encripta un archivo usando Fernet y guarda la salida.
    """
    key = load_key(key_path)
    fernet = Fernet(key)
    with open(input_path, "rb") as file:
        encrypted = fernet.encrypt(file.read())
    with open(output_path, "wb") as enc_file:
        enc_file.write(encrypted)
    print(f"Archivo encriptado: {output_path}")


def decrypt_file(input_path: str, output_path: str, key_path: str = ".local/keys/key.key"):
    """
    Desencripta un archivo usando Fernet y guarda la salida.
    """
    key = load_key(key_path)
    fernet = Fernet(key)
    with open(input_path, "rb") as enc_file:
        decrypted = fernet.decrypt(enc_file.read())
    with open(output_path, "wb") as dec_file:
        dec_file.write(decrypted)
    print(f"Archivo desencriptado: {output_path}")
