m = 4
n = 5
def ListOfLists(m,n):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = list(alphabet)
    return [[alpha[n]]*m for n in range(5)]
print(ListOfLists(m,n))
