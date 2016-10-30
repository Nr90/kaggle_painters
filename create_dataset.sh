python split_dataset.py
rm data/train_file.csv.gz
gzip data/train_file.csv
rm data/val_file.csv.gz
gzip data/val_file.csv
