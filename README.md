# Emotion Classification Project

## Overview
This project builds a machine learning system to classify emotions (sadness, joy, love, anger, fear, surprise) in text data. Using Python, it employs a pipeline of TF-IDF vectorization and Logistic Regression, achieving 90% accuracy on a dataset of 83,359 labeled text samples from Kaggle's "Emotions" dataset. The system is deployed as a real-time web app using Streamlit. Completed as of May 24, 2025.

## Features
- Classifies text into six emotions with confidence scores.
- Preprocesses text with tokenization, stopwords removal, and lemmatization.
- Includes a Streamlit app for real-time emotion prediction.
- Handles data imbalances and short text inputs effectively.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/emotion-detector.git
   cd emotion-detector
   ```
2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   - Ensure Python 3.7+ is installed.
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Note: `requirements.txt` includes `pandas`, `nltk`, `scikit-learn`, `streamlit`, `joblib`, `contractions`, `matplotlib`, `seaborn`, `kagglehub`.
4. **Download NLTK Data**:
   - Run the following in Python to download required NLTK resources:
     ```python
     import nltk
     nltk.download('punkt')
     nltk.download('stopwords')
     nltk.download('wordnet')
     nltk.download('omw-1.4')
     ```

## Usage
1. **Explore the Dataset**:
   - Run `data_exploration.ipynb` to download the Kaggle dataset and generate `emotions_with_labels.csv`.
2. **Preprocess Data**:
   - Execute `data_preprocessing.ipynb` to create `train_preprocessed.csv` and `test_preprocessed.csv`.
3. **Train the Model**:
   - Run `emotion_classification_model.ipynb` to train the model and save it as `emotion_classifier_model.joblib`.
4. **Launch the App**:
   - Start the Streamlit app:
     ```bash
     streamlit run emotion_classifier_app.py
     ```
   - Open the app in your browser (default: `http://localhost:8501`).
   - Enter text or select a sample to predict emotions (e.g., "I am so happy today!" → Joy).

## File Structure
```
emotion-detector/
│
├── .venv/                        # Virtual environment
├── data_exploration.ipynb        # Data loading and exploration
├── data_preprocessing.ipynb      # Text preprocessing
├── emotion_classification_model.ipynb  # Model training
├── emotion_classifier_app.py     # Streamlit application
├── emotion_classifier_model.joblib  # Trained model file
├── emotions_with_labels.csv      # Preprocessed dataset
├── readme.md                     # This file
├── requirements.txt              # Python dependencies
├── test_preprocessed.csv         # Test data
├── train_preprocessed.csv        # Training data
```

## Results
- **Accuracy**: 90%.
- **F1-Scores**: Sadness (0.94), Joy (0.93), Anger (0.90), Fear (0.85), Love (0.79), Surprise (0.74).
- **Confusion Matrix**: Strong performance (e.g., 22,842 correct sadness predictions), with some misclassifications (e.g., 1,103 joy as love).

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "description"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request with a detailed description.

## License
- [MIT License](LICENSE) (add a `LICENSE` file if not present).

## Acknowledgments
- Dataset: Kaggle "Emotions" by nelgiriyewithana.
- Libraries: Pandas, NLTK, Scikit-learn, Streamlit, and others.

## Contact
- For questions or feedback, use the repository issues page or contact@abdelrahman-abdelmoaty.com.
