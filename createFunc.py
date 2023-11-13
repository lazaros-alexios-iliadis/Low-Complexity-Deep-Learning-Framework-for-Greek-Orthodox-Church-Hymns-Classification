import os

newPath = "D:\\Projects\\Ymnodos\\AppliedSciences\\Mel"
if not os.path.exists(newPath):
    os.makedirs(newPath)

N = 23
for i in range(N):
    os.makedirs(os.path.join(newPath, "hymn" + str(i+1)), exist_ok=True)
