import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Canvas, Button
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from PIL import Image, ImageTk
import requests
from io import BytesIO
from ttkthemes import ThemedStyle

# Sample DataFrame with card names
df = pd.read_pickle("cards_clean_final.pkl")

# Extract card names from the DataFrame
card_names = df["name"].tolist()

reshaped_glove_embedding = np.array(df['glove_embedding'].tolist())
reshaped_glove_embedding_type_line = np.array(df['glove_embedding_type_line'].tolist())
reshaped_word2vec_embedding = np.array(df['word2vec_embedding'].tolist())
reshaped_word2vec_embedding_type_line = np.array(df['word2vec_embedding_type_line'].tolist())
reshaped_fasttext_embedding = np.array(df['fasttext_embedding'].tolist())
reshaped_fasttext_embedding_type_line = np.array(df['fasttext_embedding_type_line'].tolist())


# Create the main window
root = tk.Tk()
root.title("Card Search App")
root.configure(bg="lightblue")
label_color="lightgreen"
text_color="black"
text_style = "Arial"

style = ThemedStyle(root)
style.set_theme("black")

# Open the image file using PIL
image = Image.open("C:/Users/dcw1981/Desktop/DSC-502 Project/mtg.jpg")

# Convert the image to a format supported by Tkinter
bg_image = ImageTk.PhotoImage(image)

# Get the width and height of the image
bg_width = image.width
bg_height = image.height

# Create a canvas with the same size as the image
canvas = Canvas(root, width=bg_width, height=bg_height)
canvas.place(x=0, y=0)

# Place the image on the canvas
canvas.create_image(0, 0, image=bg_image, anchor="nw")

def update_features():
    # Update numerical features based on checkbox states
    selected_numerical_features = ['cmc'] if cmc_var.get() else []
    selected_numerical_features.extend(['power'] if power_var.get() else [])
    selected_numerical_features.extend(['toughness'] if toughness_var.get() else [])
    selected_numerical_features.extend(['U_int', 'G_int', 'W_int', 'B_int', 'R_int', 'X_int', 'C_int', 'P_int', 'S_int'] if color_intensity_var.get() else [])
    
    # Update color identity features based on checkbox state
    selected_binary_features = df[list(df.columns[23:29])] if color_identity_var.get() else []
    
    # Update text embedding features based on checkbox states
    selected_text_embeddings = []
    if glove_var.get():
        selected_text_embeddings.extend([reshaped_glove_embedding, reshaped_glove_embedding_type_line])
    if word2vec_var.get():
        selected_text_embeddings.extend([reshaped_word2vec_embedding, reshaped_word2vec_embedding_type_line])
    if fasttext_var.get():
        selected_text_embeddings.extend([reshaped_fasttext_embedding, reshaped_fasttext_embedding_type_line])
    
    return selected_numerical_features, selected_binary_features, selected_text_embeddings

# Function to handle the text changed event in the search box
def on_text_changed(event):
    # Get the current value in the search box
    current_text = entry_var.get().lower()
    # Filter the options based on the current text
    matching_options = [name for name in card_names if current_text in name.lower()]
    # Update the autocomplete options
    combobox["values"] = matching_options[:10]

# Function to handle the "GO!" button click event
def on_go_button_click():
    selected_numerical_features, selected_binary_features, selected_text_embeddings = update_features()
    
    # Check if any numerical features are selected
    if not selected_numerical_features:
        # Show a message or handle this case according to your application's logic
        print("Please select at least one numerical feature.")
        return
    
    scaler = StandardScaler()
    normalized_numerical_features = scaler.fit_transform(df[selected_numerical_features])

    # Combine all selected features
    if color_identity_var.get():  # Check if selected_binary_features is not empty
        X = np.concatenate([
            normalized_numerical_features, 
            selected_binary_features.values, 
            *selected_text_embeddings
        ], axis=1)
    else:
        X = np.concatenate([
            normalized_numerical_features, 
            *selected_text_embeddings
        ], axis=1)

    # Get the selected value from the combobox
    selected_n = int(entry_var_n.get())

    global knn
    # 5. Train KNN model
    knn = NearestNeighbors(n_neighbors=selected_n, algorithm='auto')
    knn.fit(X)

    # Get the selected value from the combobox
    selected_card = combobox.get()
    
    # Get features for the selected card
    input_card_features = get_features_for_card(selected_card, scaler, selected_numerical_features, selected_binary_features)
    
    # Find similar cards
    similar_cards = find_similar_cards(input_card_features, knn, selected_n)
    
    # Display similar card names in the terminal
    image_frame = tk.Frame(root)
    image_frame.pack()

    # Create a title for the chosen card
    chosen_card_title = tk.Label(root, text="Chosen Card", font=("Arial", 12, "bold"))
    chosen_card_title.pack()

    # Display small image of the chosen card
    chosen_card_image_frame = tk.Frame(root)
    chosen_card_image_frame.pack()
    chosen_card_image_url = similar_cards.iloc[0]['image_uris']['small']
    chosen_card_response = requests.get(chosen_card_image_url)
    chosen_card_image_data = BytesIO(chosen_card_response.content)
    chosen_card_image = Image.open(chosen_card_image_data)
    chosen_card_photo = ImageTk.PhotoImage(chosen_card_image)
    chosen_card_label_image = tk.Label(chosen_card_image_frame, image=chosen_card_photo)
    chosen_card_label_image.image = chosen_card_photo
    chosen_card_label_image.pack(side=tk.LEFT)

    # Create a title for similar cards
    similar_cards_title = tk.Label(root, text="Similar Cards", font=("Arial", 12, "bold"))
    similar_cards_title.pack()

    # Display small images of similar cards horizontally
    similar_cards_frame = tk.Frame(root)
    similar_cards_frame.pack()
    for index, card in similar_cards.iterrows():
        similar_card_image_url = card['image_uris']['small']
        similar_card_response = requests.get(similar_card_image_url)
        similar_card_image_data = BytesIO(similar_card_response.content)
        similar_card_image = Image.open(similar_card_image_data)
        similar_card_photo = ImageTk.PhotoImage(similar_card_image)
        similar_card_label_image = tk.Label(similar_cards_frame, image=similar_card_photo)
        similar_card_label_image.image = similar_card_photo
        similar_card_label_image.pack(side=tk.LEFT)


# Function to get features for a specific card
def get_features_for_card(card_name, scaler, selected_numerical_features, selected_binary_features):
    # Extract features for the given card name
    card_numerical_features = df.loc[df['name'] == card_name, df[selected_numerical_features].columns].values
    
    if color_identity_var.get():
        card_binary_features = df.loc[df['name'] == card_name, selected_binary_features.columns].values
    
    # Normalize numerical features
    card_numerical_features_normalized = scaler.transform(card_numerical_features)
    
    # Reshape embedding features
    reshaped_glove_embedding_test = np.array(df.loc[df['name'] == card_name, 'glove_embedding'].tolist())
    reshaped_glove_embedding_type_line_test = np.array(df.loc[df['name'] == card_name, 'glove_embedding_type_line'].tolist())
    reshaped_word2vec_embedding_test = np.array(df.loc[df['name'] == card_name, 'word2vec_embedding'].tolist())
    reshaped_word2vec_embedding_type_line_test = np.array(df.loc[df['name'] == card_name, 'word2vec_embedding_type_line'].tolist())
    reshaped_fasttext_embedding_test = np.array(df.loc[df['name'] == card_name, 'fasttext_embedding'].tolist())
    reshaped_fasttext_embedding_type_line_test = np.array(df.loc[df['name'] == card_name, 'fasttext_embedding_type_line'].tolist())

    selected_text_embeddings_test = []
    if glove_var.get():
        selected_text_embeddings_test.extend([reshaped_glove_embedding_test, reshaped_glove_embedding_type_line_test])
    if word2vec_var.get():
        selected_text_embeddings_test.extend([reshaped_word2vec_embedding_test, reshaped_word2vec_embedding_type_line_test])
    if fasttext_var.get():
        selected_text_embeddings_test.extend([reshaped_fasttext_embedding_test, reshaped_fasttext_embedding_type_line_test])
    

    if color_identity_var.get():
    # Concatenate features
        card_features = np.concatenate([
        card_numerical_features_normalized, 
        card_binary_features, 
        *selected_text_embeddings_test
    ], axis=1)
    else:
        card_features = np.concatenate([
        card_numerical_features_normalized, 
        *selected_text_embeddings_test
        ], axis=1)
    
    return card_features

# Function to find similar cards
def find_similar_cards(input_card_features, knnn, k=5):
    # Find most similar cards to the input card
    knn=knnn
    distances, indices = knn.kneighbors(input_card_features)
    
    # Output most similar cards
    similar_cards = df.iloc[indices[0]]
    
    return similar_cards

# Create variables to hold the state of each checkbox
cmc_var = tk.BooleanVar()
power_var = tk.BooleanVar()
toughness_var = tk.BooleanVar()
color_intensity_var = tk.BooleanVar()
color_identity_var = tk.BooleanVar()
glove_var = tk.BooleanVar()
word2vec_var = tk.BooleanVar()
fasttext_var = tk.BooleanVar()

# Create checkboxes for numerical features
cmc_checkbox = tk.Checkbutton(root, text='CMC', variable=cmc_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
cmc_checkbox.pack()
power_checkbox = tk.Checkbutton(root, text='Power', variable=power_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
power_checkbox.pack()
toughness_checkbox = tk.Checkbutton(root, text='Toughness', variable=toughness_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
toughness_checkbox.pack()
color_intensity_checkbox = tk.Checkbutton(root, text='Color Intensity', variable=color_intensity_var, command=update_features,  bg=label_color, fg=text_color, font=(text_style, 10))
color_intensity_checkbox.pack()

# Create a single checkbox for color identity
color_identity_checkbox = tk.Checkbutton(root, text='Color Identity', variable=color_identity_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
color_identity_checkbox.pack()

# Create checkboxes for text embeddings
glove_checkbox = tk.Checkbutton(root, text='Glove', variable=glove_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
glove_checkbox.pack()
word2vec_checkbox = tk.Checkbutton(root, text='Word2Vec', variable=word2vec_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
word2vec_checkbox.pack()
fasttext_checkbox = tk.Checkbutton(root, text='FastText', variable=fasttext_var, command=update_features, bg=label_color, fg=text_color, font=(text_style, 10))
fasttext_checkbox.pack()

# Create a label for the search box
label_card_name = tk.Label(root, text="Enter Card Name:", font=(text_style, 10))
label_card_name.pack()

entry_var = tk.StringVar()

# Create a combobox widget for the search box
combobox = ttk.Combobox(root, textvariable=entry_var, values=card_names, font=(text_style, 10))
combobox.pack()

# Bind the on_text_changed function to the key press event of the combobox widget
combobox.bind("<KeyRelease>", on_text_changed)


# Create a label for the n_neighbors entry
label_n = tk.Label(root, text="Enter n_neighbors:", font=(text_style, 10))
label_n.pack()

# Create a variable to hold the text in the n_neighbors entry widget
entry_var_n = tk.StringVar()

# Create an entry widget for n_neighbors
entry_n = ttk.Entry(root, textvariable=entry_var_n, font=(text_style, 10))
entry_n.pack()

# Create the "GO!" button
go_button = tk.Button(root, text="GO!", command=on_go_button_click, bg="green", fg="white", font=(text_style, 10))
go_button.pack()

# Create frames for "Chosen Card" and "Similar Cards"
chosen_card_frame = tk.Frame(root)
chosen_card_frame.pack(side=tk.LEFT, padx=10, pady=10)

similar_cards_frame = tk.Frame(root)
similar_cards_frame.pack(side=tk.RIGHT, padx=10, pady=10)


# Run the GUI
root.mainloop()

