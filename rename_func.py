import os

folder = "D:\\Projects\\Ymnodos\\AppliedSciences\\AudioDataset\\"
count = 1
for name in os.listdir(folder):
    source = folder + name
    dst = folder + "hymn" + str(count)
    os.rename(source, dst)
    count += 1

result = os.listdir(folder)
print(result)
