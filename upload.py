from flask import Flask, request
# from flask_restful import Api, Resource, reqparse
# import werkzeug
from werkzeug.utils import secure_filename

app=Flask(__name__)


@app.route('/test', methods=['POST'])
def Prediction():
    print("hello")
    file = request.files['file']
    filename = secure_filename(file.filename)
    print(file.filename)
    file.save( "./" +filename)
    return "gg"


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=6000,debug=False,threaded=True)