import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the ResNet model for feature extraction
image_model = ResNet50(weights='imagenet')
image_model = Model(image_model.input, image_model.layers[-2].output)

# Function to preprocess image and extract features
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    features = image_model.predict(img, verbose=0)
    return features

# Load image captions dataset
# Assuming captions_dict is a dictionary with {image_id: [captions]} after loading your dataset
# Example: captions_dict['123.jpg'] = ["A person riding a horse", "A man on a horse in a field"]

with open('captions_dict.pkl', 'rb') as f:
    captions_dict = pickle.load(f)

# Prepare tokenizer
captions = [caption for captions_list in captions_dict.values() for caption in captions_list]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1

# Data preparation for training
max_length = max(len(caption.split()) for caption in captions)

def create_sequences(tokenizer, max_length, caption, image):
    sequence = tokenizer.texts_to_sequences([caption])[0]
    X1, X2, y = [], [], []
    for i in range(1, len(sequence)):
        in_seq, out_seq = sequence[:i], sequence[i]
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
        X1.append(image)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Build the captioning model
def build_model(vocab_size, max_length):
    # Image feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)

    # Sequence processor model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    # Decoder model
    decoder1 = add([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Combine the models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Train the model
model = build_model(vocab_size, max_length)
# Assuming train_features and train_captions are prepared lists of image features and corresponding captions
for img_id, captions in captions_dict.items():
    image = extract_features(f"images/{img_id}")
    for caption in captions:
        X1, X2, y = create_sequences(tokenizer, max_length, caption, image)
        model.fit([X1, X2], y, epochs=1, verbose=1)

# Generate a caption for an image
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Test with a sample image
test_image = 'test_image.jpg'
photo = extract_features(test_image)
caption = generate_caption(model, tokenizer, photo, max_length)
print("Generated Caption:", caption)
plt.imshow(load_img(test_image))
plt.show()