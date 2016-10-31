rm data/train.csv
rm data/validation.csv
rm data/test.csv
rm data/manifest.csv
rm data/manifest_clean.csv
python create_manifest.py
python clean_dataset.py
python split_dataset.py
