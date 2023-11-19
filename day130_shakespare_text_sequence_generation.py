
import tensorflow as tf

#@ Preparing dataset  
shakespeare_url = "https://homl.info/shakespeare"                                                         # webpage for text 
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:                # store shakespeare_text from give url
    shakespeare_text = f.read()

print(shakespeare_text[:80])

#encoding of text
text_vec_layer = tf.keras.layers.TextVectorization(split="character",standardize="lower")      # character level encoding ,all to lowercase
text_vec_layer.adapt([shakespeare_text])                                                       # character is mapped to integer starting from 2
encoded = text_vec_layer([shakespeare_text])[0]              

encoded -= 2                                     # drop 0 of pad and 1 of unknown 
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars i.e 39
dataset_size = len(encoded)                      # total number of chars

# function that creat window like 1 window takes "hell" another take "ello" for word hello, if shuffle
def to_dataset(sequence, length, shuffle=False, seed = None, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)                                # create a tf dataset from the sequence
    ds = ds.window(length+1, shift=1, drop_remainder= True)                          # create overlapping window of length
    if shuffle:                                                                      # shuffle the dataset 
        ds = ds.shuffle(buffer_size=100_000, seed=seed)                               
    ds = ds.batch(batch_size)                                                        # batches of given size     
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)        # map window where first element last item is left and second element first item is left                                                 

length = 100                                   # length of each sequence window
tf.random.set_seed(42)                           
train_set = to_dataset(encoded[:1_000_000], length = length, shuffle= True, seed=42)    # takes first 1,000,000 element
valid_set = to_dataset(encoded[1_00_000:1_060_000],length = length)                     # 1,000,000 to 1,060,000 element as validation
test_set = to_dataset(encoded[1_060_000:], length=length)                               # after 1,060,000 for test set

#@ Building and training char RNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim =n_tokens, output_dim=16),                                                    # embedding layer encode charcter ids 
    tf.keras.layers.GRU(128, return_sequences=True),                                                           
    tf.keras.layers.Dense(n_tokens, activation="softmax")                                                             # give the probability for each character
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])                        # optimizing model
model_ckpt = tf.keras.callbacks.ModelCheckpoint("my_shakespeare_model", monitor="val_accuracy", save_best_only=True)  # checkpoint to save best model with best validation acccuracy
history = model.fit(train_set, validation_data = valid_set,epochs=10,callbacks=[model_ckpt])                          # model train and save best model by callaback function