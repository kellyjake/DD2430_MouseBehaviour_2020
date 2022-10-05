import psutil
from memory_profiler import profile

@profile
def test():
    a = []
    for i in range(100):
        for j in range(10):
            a.append(i+j)

    return a

test()

# gives a single float value
print(psutil.cpu_percent())
# gives an object with many fields
print(psutil.virtual_memory())
# you can convert that object to a dictionary 
print(dict(psutil.virtual_memory()._asdict()))
# you can have the percentage of used RAM
print(psutil.virtual_memory().percent)
print(type(psutil.virtual_memory().percent))
# you can calculate percentage of available memory
print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
