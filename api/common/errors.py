from api.common import ConflictError

class InvalidLaptopError(ConflictError):
    def __init__(self):
        super().__init__("La imagen no corresponde al modelo y marca especificados.")