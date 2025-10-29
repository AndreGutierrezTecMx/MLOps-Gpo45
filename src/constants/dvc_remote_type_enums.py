from enum import Enum

class DvcRemoteType(Enum):
    """
    Enum con los tipos de remotos DVC soportados.
    Se usa para evitar errores por strings mal escritos.
    """
    LOCAL = "local"
    GDRIVE = "gdrive"
    AZURE = "azure"
    S3 = "s3"

    @classmethod
    def list(cls):
        """Devuelve una lista con los valores válidos."""
        return [t.value for t in cls]
