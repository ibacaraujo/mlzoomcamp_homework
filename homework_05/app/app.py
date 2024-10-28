from flask import Flask, request, jsonify
import pickle

model_file = "model1.bin"
with open(model_file, 'rb') as f:
    model = pickle.load(f)

d_file = "dv.bin"
with open(model_file, 'rb') as f:
    d = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    
    X = d.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)