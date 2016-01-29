def ListsFromTuples():
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = list(alphabet)
    return [(alpha[x],alpha.index(alpha[x])+1) for x in range(26)]

def TwoLists():
    one,two = map(list,zip(*ListsFromTuples()))
    three = one,two
    return list(three)

print(TwoLists())
