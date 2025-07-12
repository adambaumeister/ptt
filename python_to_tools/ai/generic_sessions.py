from requests import Session

class BearerAuthenticatedSessionFactory:
    def __init__(self, bearer_token: str):
        """
        Basic, Bearer authenticated session factory

        Arguments:
            bearer_token {str}: Token to use
        """
        self.bearer_token = bearer_token

    def __call__(self):
        """Creates and returns a requests.Session object with the bearer token set"""
        s = Session()
        s.headers.update({"Authorization": f"Bearer {self.bearer_token}"})
        return s