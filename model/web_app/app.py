from flask import Flask, render_template, request, redirect
from predict_ import *

app = Flask(__name__, template_folder='Template')

model = init_model()

@app.route('/', methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            transcript = prediction( model, file )

    return render_template('index.html', transcript=transcript)

# run app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run( host='0.0.0.0', port=port)

# import os
# import uuid
# from flask import Flask, flash, request, redirect

# UPLOAD_FOLDER = 'files'

# app = Flask(__name__, template_folder='Template')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


# @app.route('/save-record', methods=['POST'])
# def save_record():
#     # check if the post request has the file part
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     # if user does not select file, browser also
#     # submit an empty part without filename
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
#     file_name = str(uuid.uuid4()) + ".wav"
#     full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#     file.save(full_file_name)
#     return '<h1>Success</h1>'


# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug = True, threaded=True)
