import numpy as np
import csv
import matplotlib.pyplot as plt
csvfile = open('data/europa/Europa hibrido power on aerofolio.csv', mode='r', newline='')
print(csvfile)


notas = open('notas.csv', 'w', newline='')
obj = csv.writer(notas)
for i in csvfile:
    obj.writerow(i)
