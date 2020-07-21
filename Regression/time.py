import time

start = time.time()

for i in range(10000000):
    print(i)

end = time.time()
print(end-start)