import pickle
import flask

app = flask.Flask("__main__")

#loading trained model
model = pickle.load(open("SVM Training And Exporting/model.pkl","rb"))

@app.route('/predict', methods=['POST'])
def predict():
	feature_array = request.get_json()['feature_array']
	print(feature_array)
	prediction = model.predict([feature_array]).tolist()
	response = {}
	response['predictions'] = prediction

	return flask.jsonify(response)
