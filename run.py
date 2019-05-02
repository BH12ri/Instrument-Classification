import pickle
import os
import flask
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import AudioFeaturesExtraction.SingleAudioFeaturesExtraction.getFeatures as getFeature
app = flask.Flask(__name__)

# Loading trained model
model = pickle.load(open("SVM Training And Exporting/model.pkl","rb"),encoding='latin1')

# All audio files uploaded to same directory
UPLOAD_FOLDER = '/home/avinash/Desktop/GitHub/Audio_Processing/UPLOAD'
ALLOWED_EXTENSIONS = set(['wav']) # Only all files with wav extension is allowed to submitted

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Setting the upload folder path


# Checking files is in the correct extension format
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analyze/', methods=['POST','GET'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			feature = getFeature.getFeature(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			pre = model.predict([feature['features']])[0]
			if(pre == 0):
				pre = 'Violin'
			elif(pre == 1):
				pre = 'Flute'
			elif(pre == 2):
				pre = 'Piano'
			else:
				pre = 'Trumpet'
			return jsonify(pre)
				
			
			
			
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
	<input type=file name=file>
	<input type=submit value=Upload>
	</form>
	'''

if(__name__ == '__main__'):
	app.run()
