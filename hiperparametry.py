# Hiperparametry do projektu pix2pix
n_epochs = 250
batch_size = 64
lr = 1e-4
loss_level = 10  # współczynnik przy L1 loss
w_decay = 1e-4      # współczynnik regularyzacji dla optymalizatorów
is_show = False  # czy pokazywać wyniki testowe na koniec

# Split danych
full_size = 15922
num_holdout = 10  # liczba ostatnich zdjęć do pominięcia
test_split = 0.2
usable_size = full_size - num_holdout
test_size = int(usable_size * test_split)
train_size = usable_size - test_size
train_start = 0
train_stop = train_size
test_start = train_size
test_stop = usable_size

# ścieżki do folderów
path1 = r"C:\Users\RODO\Desktop\gan-in-pixel-arts\0_0"
path2 = r"C:\Users\RODO\Desktop\gan-in-pixel-arts\2_2"
input_path = r"C:\Users\RODO\Desktop\nowy_image.png"
output_path = r"C:\Users\RODO\Desktop\nowy_image_gen.png"


