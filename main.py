from flask import Flask
import detect_logo
app = Flask(__name__)

@app.route('/')
def hello_world():
  try:
    predictor = detect_logo.main('test.jpg')
  except Exception as e:
    return e
  return predictor + "from yury"

if __name__ == '__main__':
  app.run()
