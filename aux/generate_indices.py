
import random

def main():
    howMany = 1001
    numbers = []

    infile = open ('generated.txt', 'w')

    for n in range(1,howMany):
        index = '\nimg_' + str(n)
        infile.write(index)
    infile.close()

main()