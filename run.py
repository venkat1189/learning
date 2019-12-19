from flask import Flask


app = Flask(__name__)

@app.route('/app')
def first_app():
    print('Hello venkat...This app is executed')
    return "Hello World..... This app successfully deployed to Elasticbeanstack using Codepipeline nd ECR"



if __name__ == '__main__':
    app.run(host = '0.0.0.0')