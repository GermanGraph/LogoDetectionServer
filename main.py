from flask import Flask
import detect_logo
app = Flask(__name__)

@app.route('/')
def hello_world():
  predictor = detect_logo.main('test.jpg')
  return predictor

if __name__ == '__main__':
  app.run()
