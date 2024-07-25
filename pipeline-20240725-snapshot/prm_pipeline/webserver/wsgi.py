import os

from .app import create_app

app = create_app(debug=bool(os.environ.get("DEBUG", False)))


@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST")
    return response


if __name__ == "__main__":
    port_number = int(os.environ.get("WEBSERVER_PORT", 25565))
    app.run(port=port_number)
