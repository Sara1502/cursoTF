import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


sentences = [
    "I like eggs and ham.",
    "I love chocolate and bunnies.",
    "I hate onions."
]

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)
print(tokenizer.word_index)

data = pad_sequences(sequences)
print(data)

MAX_SEQUENCE_LENGHT = 5
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGHT)
print(data)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGHT, padding='post')
print(data)

data = pad_sequences(sequences, maxlen=6)
print(data)

data = pad_sequences(sequences, maxlen=4)
print(data)

data = pad_sequences(sequences, maxlen=4, truncating='post')
print(data)