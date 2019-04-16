import timeit
from sklearn import datasets
from randomforest import randomforest

def main():
    digits = datasets.load_digits()
    data = digits['data']
    target  = digits['target']
    predictor = randomforest(data, target)
    time = timeit.timeit(predictor.run, number=10)
    stats = predictor.run()
    print(f' Classifier randomforest \n ------------------\n')
    print(f' Whole execution process lasted {time:.2f} seconds (mean of 10 executions)\n')
    print(stats)


if __name__ == '__main__':
    main()