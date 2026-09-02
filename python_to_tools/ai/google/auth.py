from google.auth import default
import google.auth.transport.requests
from requests import Session


class GoogleNativeSessionFactory:
    def __init__(
            self, scopes: list = None
    ):
        self.scopes = scopes
        if not scopes:
            self.scopes = [
                "https://www.googleapis.com/auth/cloud-platform"
            ]

    def get_credentials(self):
        """Get credentials from gcloud using the app default credentials"""
        credentials, _ = default(scopes=self.scopes)
        credentials.refresh(google.auth.transport.requests.Request())

        return credentials

    def __call__(self):
        """Creates and returns a requests.Session object with the bearer token set. In this way we are using the
        google provided credentials as the token itself."""

        s = Session()
        credentials = self.get_credentials()
        s.headers.update({"Authorization": f"Bearer {credentials.token}"})
        return s

    @property
    def bearer_token(self):
        """Return just the token, not the session, useful for things like the OpenAI client library"""
        return self.get_credentials().token

