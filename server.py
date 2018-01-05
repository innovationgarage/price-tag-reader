from flask import Flask, Response, request
import os.path
import labeler
import labelreader
import uuid
import json

app = Flask(__name__, static_folder=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        ext = os.path.splitext(file.filename)[1]
        path = os.path.join("uploads", "%s%s" % (uuid.uuid4(), ext))
        file.save(path)
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
