from flask import Flask

app = Flask(__name__)

@app.route('/app')
def test():
    print('Enters into app')
    return "Hi There"


if __name__=="__main__":
    app.run(host='0.0.0.0')