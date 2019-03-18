import pickle 
import numpy  as np
import math
import matplotlib.pyplot as plt

import sys

K_NUM = [1,2,3,4,5,10,15,20,25,30,35,40,45,50,55,60,80,100,120,140,160,180,200]

class myknn_regressor ():
    
    def __init__(self, k=20, filters='remove_outliers'):
        self.k = k
        if k < 10:
            self.filters = 'equal_weight'
        else:
            if filters == 'equal_weight' or filters=='remove_outliers':
                self.filters = filters
        self.x_train = []
        self.y_train = []
        self.distri = dict() #distribution

    def fit(self, x_train, y_train):
        feature = dict()
        length = len(x_train)
        fea_length = len(x_train[0])

        for data in x_train:
            index = 0 
            for detail in data:
                if not feature.get(index):
                    feature[index] = []
                feature[index].append(detail)
                index += 1
        for key, val in feature.items():
            means = np.mean(val)
            std = np.std(val, dtype=np.float64)
            self.distri[key] = {'mean': means, 'std': std}
            result = []
            for i in val:
                result.append((i-means)/std)
                
            feature[key] = result
        x_std_train = []
        for i in range(length):
            temp = []
            for j in range(fea_length):
                temp.append(feature[j][i])
            x_std_train.append(temp)
        self.x_train = x_std_train

        self.y_train = y_train

    def predict(self, x_test):
        temp = x_test.copy()
        for i in range(len(x_test)):
            temp[i] = (x_test[i]- self.distri[i]['mean']) / self.distri[i]['std']
        record = dict()
        for i in range(len(self.x_train)):
            record[i] = np.linalg.norm(self.x_train[i] - temp)
        result = sorted(record.items(), key=lambda d: d[1])
        
        near_k = result[:self.k]
        if self.filters == 'remove_outliers':
            val = []
            for i in self.get_val(near_k):
                val.append(self.y_train[i])
            Q1 = np.quantile(val, 0.25)
            Q3 = np.quantile(val, 0.75)
            IQR = Q3 - Q1 
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            for i in range(self.k):
                if self.y_train[near_k[i][0]] > upper_bound or self.y_train[near_k[i][0]] < lower_bound:
                    near_k[i] = None

        avg_y = 0
        count = 0
        for i in range(self.k):
            if near_k[i] != None:
                avg_y += self.y_train[near_k[i][0]]
                count += 1
        return avg_y / count

    def get_val(self, record):
        ans = []
        for i in record:
            ans.append(i[0])
        return ans

def  main():
    with open('msd_data1.pickle', 'rb') as f:
        data = pickle.load(f)
        x_axis = []
        y_axis = []
        for k in K_NUM:
            myknn = myknn_regressor(1, 'remove_outliers')
            myknn.fit(data['X_train'], data['Y_train'])
            RSME = 0
            # num = 10
            num = len(data['X_test'])
            for i in range(num):
                print(i)
                y_pred = myknn.predict(data['X_test'][i]) 
                RSME += ((y_pred - data['Y_test'][i]) **2)
            RSME = RSME / num
            print("RSME :" ,math.sqrt(RSME))
            x_axis.append(k)
            y_axis.append(RSME)
            break
        #     break
        # with open ('ans.txt', 'w') as f2:
        #     for i in y_axis:
        #         f2.write(str(i))
        plt.plot(x_axis, y_axis, 'ro')
        plt.ylabel('RMSE')
        plt.xlabel('k number')
        plt.show()
    
if __name__ == '__main__':
    main()

