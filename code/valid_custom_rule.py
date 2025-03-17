def valid_rules(x, y):
    flag = []
    tolerance = 1e-5
    for i in range(len(x)):
        if abs(x[i]-y[i]) <= 0.2 + tolerance:
            flag.append(2)
        elif x[i] > 1 and y[i] > 1 and (x[i]-1.2)*(y[i]-1.2) > 0:
            flag.append(1)
        elif x[i] < 1 and y[i] < 1 and (x[i]-0.8)*(y[i]-0.8) > 0:
            flag.append(1)
        else:
            flag.append(0)
    return flag


if __name__ == '__main__':
    valid_rules([0.8], [0.8])
