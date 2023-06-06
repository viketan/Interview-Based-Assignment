from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)

# Load the trained model
model = joblib.load("genre_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler =joblib.load("scaler.pkl")

def preprocessing(input):
    return scaler.transform(input)

def decode(input):
    return encoder.inverse_transform(input)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input values from the form
    tempo = float(request.form.get("tempo"))
    beats = float(request.form.get("beats"))
    chroma_stft = float(request.form.get("chroma_stft"))
    rmse = float(request.form.get("rmse"))
    spectral_centroid = float(request.form.get("spectral_centroid"))
    spectral_bandwidth = float(request.form.get("spectral_bandwidth"))
    rolloff = float(request.form.get("rolloff"))
    zero_crossing_rate = float(request.form.get("zero_crossing_rate"))
    mfcc1 = float(request.form.get("mfcc1"))
    mfcc2 = float(request.form.get("mfcc2"))
    mfcc3 = float(request.form.get("mfcc3"))
    mfcc4 = float(request.form.get("mfcc4"))
    mfcc5 = float(request.form.get("mfcc5"))
    mfcc6 = float(request.form.get("mfcc6"))
    mfcc7 = float(request.form.get("mfcc7"))
    mfcc8 = float(request.form.get("mfcc8"))
    mfcc9 = float(request.form.get("mfcc9"))
    mfcc10 = float(request.form.get("mfcc10"))
    mfcc11 = float(request.form.get("mfcc11"))
    mfcc12 = float(request.form.get("mfcc12"))
    mfcc13 = float(request.form.get("mfcc13"))
    mfcc14 = float(request.form.get("mfcc14"))
    mfcc15 = float(request.form.get("mfcc15"))
    mfcc16 = float(request.form.get("mfcc16"))
    mfcc17 = float(request.form.get("mfcc17"))
    mfcc18 = float(request.form.get("mfcc18"))
    mfcc19 = float(request.form.get("mfcc19"))
    mfcc20 = float(request.form.get("mfcc20"))
    # Create a list of input values
    input_data = [tempo, beats, chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfcc1, mfcc2, mfcc3, mfcc4, mfcc5, mfcc6, mfcc7, mfcc8, mfcc9, mfcc10, mfcc11, mfcc12, mfcc13, mfcc14, mfcc15, mfcc16, mfcc17, mfcc18, mfcc19, mfcc20]  
    input_data = np.array(input_data)
    #Preprocessing
    input_data = preprocessing([input_data])
    # Make a prediction using the loaded model
    genre = model.predict(input_data)
    genre = decode(genre)[0]
    return render_template("result.html", genre=genre)

if __name__ == "__main__":
    app.run(debug=True)
