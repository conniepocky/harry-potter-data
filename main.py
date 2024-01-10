import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

# Read in the data
data = pd.read_csv("data/script1.csv", sep=";")

#remove whitespace from characternames in data

data["Character"] = data["Character"].str.strip()

#find how many lines each character has
linesPerCharacter = data.groupby("Character").size().sort_values(ascending=False)

print(linesPerCharacter)

#find the top 15 characters

top10 = linesPerCharacter.head(15)

plt.bar(top10.index, top10.values)

plt.xticks(rotation=90, fontsize=7)

plt.show()