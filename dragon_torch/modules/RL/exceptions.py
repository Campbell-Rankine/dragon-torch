class InvalidBackendException(Exception):
    def __init__(self, backend_type, supported_backends, message: str = "not found in"):
        super().__init__(message)
        self.backend_type = backend_type
        self.supported_backends = supported_backends

    def __repr__(self):
        return f"{self.backend_type} {self.args[0]} {self.supported_backends}"