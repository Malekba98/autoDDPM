filename = "./data/ATLAS/png_128_sl_95/Atlas_train_/Atlas_train_6.png"

# Split the filename on underscore and take the last element, then remove the .png extension
index = filename.split("_")[-1].replace(".png", "")

print(filename.split("_"))
# Convert the index to an integer
index = int(index)

print(index)
