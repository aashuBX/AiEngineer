class Neo4jClient:
    def __init__(self, uri, username, password):
        self.uri = uri
        self.username = username
        self.password = password
        
    def connect(self):
        pass
        
    def execute_query(self, query: str, parameters: dict = None):
        return []
