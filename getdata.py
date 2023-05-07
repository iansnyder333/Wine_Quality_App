import pandas as pd
import numpy as np

class GetData(object):
    def __init__(self, redfile, whitefile):
        self.reds = self.prep(pd.read_csv(redfile), 'red')
        self.whites = self.prep(pd.read_csv(whitefile),'white')
        self.all = self.reds.append(self.whites, ignore_index=True)

    def prep(self, data, flavor):
        dic = {}
        for c,v in data.items():
            colum = c.split(';')
            col=[]
            for n in colum:
                n = n.strip('\"')
                col.append(n)
                dic[n] = [] 

            for val in v:
                r = val.split(';')
                for ind,j in enumerate(r):
                    dic[col[ind]].append(j)  
          
        df = pd.DataFrame(dic)
        df['type'] = [flavor for i in range(len(df))]
        df['grade'] = [1 if int(i)>6 else 0 for i in df['quality']]
        return df
    
        
    
        
      



def main():
    wines = GetData('data/winequality-red.csv', 'data/winequality-white.csv')
    print(wines.all.head())
if __name__=='__main__':
    main()
