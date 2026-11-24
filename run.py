from backend.app import app

if __name__ == "__main__":
    # Local-only development server; images and annotations stay on this machine.
    app.run(host="127.0.0.1", port=8000, debug=False)
