def my_fun(g):
    print(g)


for i in range(10):
    my_fun(i)
    if i > 0:
        my_fun(i)
        print(f'Это замечательный текст будет напечатан много раз {i}')
