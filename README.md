# 🌟 Urdu Word Embeddings 📚

This project demonstrates the use of Word2Vec models to create word embeddings for Urdu language text. Word embeddings are dense vector representations of words that capture semantic and syntactic relationships between words. These embeddings can be used for various natural language processing tasks such as word similarity, language generation, and sentiment analysis. 💪

## 📋 Dataset

The project utilizes the "urdu short 20k.txt" news dataset, which contains a collection of 20,000 lines of Urdu language text. This dataset serves as a starting point for experimentation.  
📚 However, for better accuracy, users can replace it with a larger dataset containing more than 100,000 lines of text. Adding more data can improve the quality of word embeddings and the performance of the LSTM model.  

## 🧠 Word2Vec Model Training

The Word2Vec model is trained using the Continuous Bag of Words (CBOW) and Skip-Gram methods. The trained models generate word embeddings by considering the context of words in a given sentence. The model parameters, such as vector size, window size, and minimum count, can be adjusted based on the specific requirements. 📖

## 🔎 Word Similarity and Visualization

The trained Word2Vec models allow for calculating word similarity and finding similar words. The similarity between two words is calculated using the cosine similarity measure. 📊 The project provides code for visualizing word similarity using bar charts and heatmaps.

## 📊 LSTM Model for Text Classification

In addition to word embeddings, the project includes an LSTM (Long Short-Term Memory) model for text classification. The model is trained on the tokenized Urdu text and corresponding labels. The model architecture consists of an embedding layer, LSTM layer, dropout layer, and dense layer. By default, the model is trained for a certain number of epochs (e.g., 1).

📝 For users who want to improve the accuracy of the LSTM model, it is recommended to run the training for a higher number of epochs. Increasing the number of epochs (e.g., 5 or 10) allows the model to learn more from the data and potentially improve its performance.

## ⚙️ Requirements

The project requires the following dependencies:

- NLTK
- Gensim
- scikit-learn
- Pandas
- NumPy
- TensorFlow

Make sure to install these dependencies before running the code. 🛠️

## 🚀 Usage

1. Clone the repository: `git clone https://github.com/your-username/urdu-word-embeddings.git`
2. Navigate to the project directory: `cd urdu-word-embeddings`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the code: `python main.py`

Note: Additional NLTK resources may need to be downloaded. Uncomment the `nltk.download()` lines in the code and follow the instructions. 📥

## 🤝 Contribution

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. 👍

## 📜 Acknowledgements

The "urdu short 20k.txt" dataset used in this project is obtained from [urdu short 20k.txt](https://drive.google.com/file/d/16Sg5NDB-rr6ROrAee8TUWMTDiyGzOf2b/view?usp=sharing) and is used for research purposes.  
If you want to further improve the accuracy, Similarity so you can use this dataset that contain more than 100,000 lines of urdu news data  [urdu_news_dataset.txt](https://drive.google.com/file/d/1kueVp6_YnO5osmYrX4RzdIPzq9Hf3Aq4/view?usp=sharing)  
📖The project is built using the NLTK, Gensim, scikit-learn, Pandas, NumPy, and TensorFlow libraries. 🧰

## 📚 References

Mikolov, T., et al. "Efficient Estimation of Word Representations in Vector Space." arXiv preprint arXiv:1301.3781 (2013). 📖
