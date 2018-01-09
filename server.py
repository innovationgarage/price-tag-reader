import os.path
import uuid
import json
from binascii import a2b_base64

from flask import Flask, Response, request, flash, redirect

import labeler
import labelreader


app = Flask(__name__, static_folder=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    data_uri = request.form.get('data')
    if not data_uri:
        flash('No selected file')
        return redirect(request.url)

    info, data = data_uri.split(',')
    _, info = info.split(":")
    mimetype, encoding = info.split(";")
    if mimetype != "image/png" or encoding != "base64":
        flash('Photo is in wrong format')
        return redirect(request.url)

    binary_data = a2b_base64(data)
    path = os.path.join("uploads", "%s.%s" % (uuid.uuid4(), "png"))

    with open(path, 'wb') as image_file:
        image_file.write(binary_data)

    return Response(json.dumps([{"bbox": bbox, "texts": labelreader.mergeLineBoxesAndtexts(lineboxes, linetexts)}
                                for bbox, image, lineboxes, linetexts, borders, objgrad in labeler.readLabels(path)],
                                ensure_ascii=False).encode("utf-8"),
                    mimetype="application/json")

@app.route('/')
@app.route('/<path:path>')
def get_resource(path = ''):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    path = os.path.join("static" , path)
    if os.path.isdir(path):
        path = os.path.join(path, "index.html")
    with open(path) as f:
        content = f.read()
    return Response(content, mimetype=mimetype)

if not os.path.exists("uploads"):
    os.mkdir("uploads")

app.run(host="localhost", port=8080)
