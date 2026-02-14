def tek_sayilar(n):
    for i in range(n):
        if i%2!=0:
            yield i

g = tek_sayilar(100)

for x in g:
    print(x)