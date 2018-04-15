import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, send_file
from flask.json import jsonify
import detect_logo
import os
import io
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "images"
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def bad_request(reason):
    response = jsonify({"error": reason})
    response.status_code = 400
    return response


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return bad_request("empty body")
        file = request.files['file']
        if file.filename == '':
            return bad_request("empty filename")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global graph_params, sess
            detect_logo.main(
                filename, graph_params=graph_params, sess=sess)
            # return send_file(io.BytesIO(img.read()), mimetype='image/jpg')
            return send_file(os.path.join("results", filename))
    except Exception as e:
        bad_request(e)


@app.route('/')
def hello_world():
    filename = "test.jpg"
    global graph_params, sess
    detect_logo.main(
        filename, graph_params=graph_params, sess=sess)
    # return send_file(io.BytesIO(img.read()), mimetype='image/jpg')
    return send_file(os.path.join("results", filename))


if __name__ == '__main__':
    graph_params, sess = detect_logo.init_tf()
    app.run()
