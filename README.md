# 🚖 Uber Trip Base Predictor

This Streamlit app predicts the base location for Uber trips based on various input features such as latitude, longitude, weekday, and hour. The app uses a machine learning model trained on historical Uber trip data and provides insightful visualizations to better understand the model's performance.

## 🚀 Features

- **Base Location Prediction**: Input trip details to get the predicted base location.
- **Confusion Matrix Visualization**: Visualize the confusion matrix of the model on the test dataset to evaluate performance.
- **Feature Importance**: Display the importance of each feature used by the model in making predictions.

## 🛠️ Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/uber-trip-base-predictor.git
    cd uber-trip-base-predictor
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # For Windows
    source .venv/bin/activate  # For macOS/Linux
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## 📋 Usage

1. **Run the Streamlit app:**

    ```bash
    streamlit run uber.py
    ```

2. **Open your web browser** and go to `http://localhost:8501`.

3. **Input the trip details** such as latitude, longitude, weekday, and hour in the provided fields.

4. **Click the "Predict Base Location" button** to see the predicted base location.

## 📁 Project Structure

- `uber.py`: The main Streamlit app script.
- `train_model.py`: Script for training the machine learning model.
- `uber_data.csv`: The dataset containing Uber trip details.
- `README.md`: Project documentation.
- `LICENSE`: License information.

## 🔍 Data Preprocessing

The following preprocessing steps are applied to the data:
- **Handling missing values**.
- **Converting categorical features** to numerical values.
- **Feature scaling** where necessary.

## 📈 Model

The app uses a **RandomForestClassifier** to predict the base location of Uber trips. The model is trained on a subset of the data and evaluated on a separate test set.

## 🖼️ Visualizations

- **Confusion Matrix**: Displays the model's performance on the test set.
- **Feature Importance**: Shows the importance of each feature in predicting the base location.

## 🧪 Example

1. **Input trip details**: Latitude: `40.7128`, Longitude: `-74.0060`, Weekday: `3`, Hour: `14`
2. **Click "Predict Base Location"**
3. **Output**: "The predicted base location is: `B02512`"

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

## 🌟 Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct and the process for submitting pull requests.

## 📧 Contact

For any questions, feel free to reach out:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://www.linkedin.com/in/yourprofile/)
- **GitHub**: [Your GitHub](https://github.com/yourusername/)

---

*This project was developed to demonstrate the capabilities of predicting base locations for Uber trips using machine learning and interactive visualizations with Streamlit.*

![App Screenshot](screenshot.png)
