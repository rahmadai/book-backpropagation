import csv
import json

csvfile = open('dataset/set.csv', 'r')

jsonArray = []
fieldnames = ("No","Judul","Penulis","Genre","Tahun Terbit","Kategori")
reader = csv.DictReader( csvfile, fieldnames)
for row in reader:
    # json.dump(row, jsonfile)
    # jsonfile.write('\n')
    jsonArray.append(row)

print(json.dumps(jsonArray))