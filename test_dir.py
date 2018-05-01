from os import listdir
from os.path import isfile, join
import glob

mypath = "~/Documents/neural-style-master/frames"
mypath = "frames"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for infile in sorted(glob.glob('frames//*'), key=numericalSort):
    print ("Current File Being Processed is: " + infile)
source_files = sorted(glob.glob('frames//*'), key=numericalSort)
for i in range(len(sorted(glob.glob('frames//*'), key=numericalSort))):
	print (i)
output_files = []
for i in range(len(source_files)):
        output_files.append('Rnad' + '/' + str(i) + '.jpg')

print(output_files)
print(onlyfiles)
print(output_files[5])