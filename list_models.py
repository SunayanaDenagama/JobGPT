import os
from dotenv import load_dotenv
from gemini_service import API_VERSIONS, REQUEST_ERRORS, discover_text_models, make_client

load_dotenv()

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment or .env first.")
    for version in API_VERSIONS:
        try:
            with make_client(api_key, version) as client:
                names = discover_text_models(client)
            print(f"{version}: {', '.join(names) or 'no supported Flash models'}")
        except REQUEST_ERRORS as error:
            print(f"{version}: listing failed ({getattr(error, 'code', type(error).__name__)})")
