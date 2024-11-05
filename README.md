# NLP Text Classifier

This project implements a text classification model using TensorFlow. The model preprocesses text data, removes stopwords, and trains a neural network to classify text into predefined categories.

## Project Structure


data/
    articles.csv
main.ipynb


- articles.csv: Contains the dataset used for training and validation.
- main.ipynb: Jupyter notebook containing the code for preprocessing, training, and evaluating the text classification model.

## Setup

1. Clone the repository.
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing

The fit_vectorizer function defines and adapts a text vectorizer to preprocess the text data by removing stopwords and punctuation, and converting the text to lowercase.


def fit_vectorizer(train_sentences, standardize_func):
    '''
    Defines and adapts the text vectorizer

    Args:
        train_sentences (tf.data.Dataset): sentences from the train dataset to fit the TextVectorization layer
        standardize_func (FunctionType): function to remove stopwords and punctuation, and lowercase texts.
    Returns:
        TextVectorization: adapted instance of TextVectorization layer
    '''
    
    vectorizer = tf.keras.layers.TextVectorization( 
        standardize="lower_and_strip_punctuation",
        max_tokens=VOCAB_SIZE,
        output_sequence_length=max_length,
    ) 
    
    train_sentences = train_sentences.map(lambda x: standardize_func(x))
    
    vectorizer.adapt(train_sentences)
    
    return vectorizer


### Training

The notebook includes functions to split the dataset into training and validation sets, preprocess the data, and train the model using different hyperparameters.

### Evaluation

The model's performance is evaluated using accuracy and loss metrics. The notebook includes functions to plot these metrics and visualize the learning rate vs. loss and accuracy.

## Running the Notebook

1. Open main.ipynb in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to preprocess the data, train the model, and evaluate its performance.

 ## License

This project is licensed under the MIT License. See the LICENSE file for details.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
