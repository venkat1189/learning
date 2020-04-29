from flask import Flask, request

app = Flask(__name__)

@app.route('/app')
def greetings():
	name = request.args.get('name')
	return "Hi {}, How are you".format(name)

def create_app():
	return app

