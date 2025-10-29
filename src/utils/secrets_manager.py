import os
import json
import logging
from cryptography.fernet import Fernet
from pathlib import Path

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Clase para manejar claves y secretos cifrados con Fernet.
    Compatible con la versión funcional que usas actualmente.
    """

    def __init__(
        self,
        key_path: str = ".local/keys/key.key",
        secrets_path: str = "configs/secrets_encrypted.json"
    ):
        self.key_path = Path(key_path)
        self.secrets_path = Path(secrets_path)

    def generate_key(self) -> bytes:
        """
        Genera una nueva clave Fernet y la guarda en el archivo especificado.
        """
        try:
            os.makedirs(self.key_path.parent, exist_ok=True)
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as key_file:
                key_file.write(key)
            logger.info(f"✅ Nueva key generada en {self.key_path}")
            return key
        except Exception as e:
            logger.exception("❌ Error al generar la key")
            raise

    def load_key(self) -> bytes:
        """
        Carga la clave Fernet desde el archivo especificado.
        """
        if not self.key_path.exists():
            raise FileNotFoundError(f"No se encontró la clave en {self.key_path}")
        with open(self.key_path, "rb") as key_file:
            return key_file.read()

    def encrypt_file(self, input_path: str, output_path: str):
        """
        Encripta un archivo usando Fernet y guarda la salida.
        """
        key = self.load_key()
        fernet = Fernet(key)
        with open(input_path, "rb") as file:
            encrypted = fernet.encrypt(file.read())

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as enc_file:
            enc_file.write(encrypted)

        logger.info(f"🔐 Archivo encriptado: {output_path}")

    def get_secret(self, token: str):
        """
        Desencripta el archivo de secrets y devuelve el valor del token esperado.
        """
        if not self.secrets_path.exists():
            raise FileNotFoundError(f"No se encontró {self.secrets_path}")

        key = self.load_key()
        fernet = Fernet(key)
        with open(self.secrets_path, "rb") as f:
            decrypted = fernet.decrypt(f.read())

        secrets = json.loads(decrypted.decode())
        if token not in secrets:
            raise KeyError(f"El token '{token}' no existe en el archivo de secretos.")
        return secrets[token]

    def get_all(self) -> dict:
        """
        Devuelve todos los secretos desencriptados como diccionario.
        """
        key = self.load_key()
        fernet = Fernet(key)
        with open(self.secrets_path, "rb") as f:
            decrypted = fernet.decrypt(f.read())
        return json.loads(decrypted.decode())
