from flask import Flask

application = Flask(__name__)

@application.route('/app')
def test():
    print('Enters into app')
    return "Hi There"


if __name__=="__main__":
    application.run(host='0.0.0.0')
