from pyspark.mllib.regression import LabeledPoint,LassoWithSGD
import time
def parsePoint(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return LabeledPoint(values[0], values[1:])
datas=[]
for i in range(1,8):
    path = "/Users/dongchunfeng/Desktop/data_spark/data_" + str(i)
    data = sc.textFile(path)
    datas.append(data.map(parsePoint))
start_time = time.time()
for parsedData in datas:
    for reg in [0.01,0.05,0.1,0.5,1,5,10]:
        model = LassoWithSGD.train(parsedData, iterations=100, step=0.000001,regParam=reg)
        valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        MSE = valuesAndPreds \
            .map(lambda (v, p): (v - p)**2) \
            .reduce(lambda x, y: x + y) / valuesAndPreds.count()
        print("Mean Squared Error = " + str(MSE))
print("--- %s seconds ---" % (time.time() - start_time)