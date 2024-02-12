import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

class CVResultAnalyzer():
    def __init__(self, targets, pickle_files, score='f1', n_splits=5):

        # loop over each target and pickle_file
        for ijk, (target, filename) in enumerate(zip(targets, pickle_files)):

            # open the file
            with open(filename, 'rb') as fid:
                gclf = pickle.load(fid)
            result = gclf.cv_results_

            # parse this cv_results
            params = result['params']
            cols = defaultdict(list)
            for param in params:
                for k,v in param.items():
                    cols[k].append(str(v))
            cols['mean_fit_time'] = list(result['mean_fit_time'])
            for isplit in range(n_splits):
                cols[f'split{isplit}_test'] = list(result[f'split{isplit}_test_{score}'])
            cols['mean_test'] = list(result[f'mean_test_{score}'])
            cols['std_test'] = list(result[f'std_test_{score}'])

            # turn into df
            df = pd.DataFrame.from_dict(cols)
            df['target'] = target
            if ijk == 0:
                self.result_df = df.copy()
            else:
                self.result_df = pd.concat((self.result_df, df), axis=0)

        # this is useful in scatterplot
        self.result_df.reset_index(names=['id'], inplace=True)
        
    def filter(self, conds):
        ''' 
           filter according to conds
           every condition is (col, vals)
        '''
        if len(conds) == 0:
            return self.result_df.copy()
        for i, (col, vals) in enumerate(conds):
            tmp = self.result_df[col].isin(vals)
            if i == 0:
                cond = tmp
            else:
                cond = cond & tmp
        return self.result_df[cond].copy()
    
    def _get_xticks(self, df, col):
        tmp = df.groupby('rank')[col].first()
        ticks = list(tmp.index)
        labels = tmp.values
        ticks_keep, labels_keep = [ticks[0]], [labels[0]]
        for i in range(1, len(ticks)):
            if labels[i] != labels[i-1]:
                ticks_keep.append(ticks[i])
                labels_keep.append(labels[i])
        return ticks_keep, labels_keep
    
    def scatterplot(self, target, hue, ys, x_main, x_secondary=None, figsize=(8,6), conds=[]):
        ''' 
          make a scatter plot for one target, with
            each subplot is the score from ys
            colored by hue            
            xaxis is marked by x_main
        '''
        # pick out this target
        df = self.filter([('target',[target])] + conds)
        
        # assemble the plot for each y
        fig, axs = plt.subplots(len(ys),1, figsize=figsize)
        axs = axs.flatten()
        df['rank'] = df.groupby(hue)['id'].rank()
        for i, (y, ax) in enumerate(zip(ys, axs)):
            sns.scatterplot(data=df, y=y, x='rank', hue=hue, ax=ax)
            if i == 0:
                ax.legend(loc='upper right', bbox_to_anchor=(1.2,1))
            else:
                ax.get_legend().remove()
            ax.xaxis.set_tick_params(labelbottom=False)
        
        # make title and labels
        ticks, labels = self._get_xticks(df, x_main)
        axs[-1].set_xticks(ticks, labels, rotation=90)
        axs[-1].xaxis.set_tick_params(labelbottom=True)
        axs[-1].set_xlabel(x_main)
        
        if x_secondary:
            ticks, labels = self._get_xticks(df, x_secondary)
            axs[0].set_xticks(ticks, labels, rotation=90)
            axs[0].set_xlabel(x_secondary)
            axs[0].xaxis.set_tick_params(labeltop=True)            
        
        axs[0].set_title(target + ': ' + hue)
        plt.show()
    
    def display_topk(self, k, conds=[]):
        ''' display topk in each target '''
        return self.filter(conds).groupby('target').\
               apply(lambda x: x.sort_values(by='mean_test', ascending=False).iloc[:k])

    def pivot_one_param(self, conds, param, index, val):
        '''
          show the val for one param when all other params are fixed
        '''
        return pd.pivot(self.filter(conds=conds), columns=param, index=index, values=val)
    
    def barplot(self, tests):
        '''
            make a barplot of fit for different tests
            each test is (test_name, conds)
        '''
        output = pd.DataFrame()
        for i, (test_name, conds) in enumerate(tests):
            df = self.filter(conds)
            tmp = df.groupby('target').agg({'mean_test':'max'}).reset_index(names=['target'])
            tmp.columns = ['target', test_name]
            if i == 0:
                output = tmp.copy()
            else:
                output = pd.merge(left=output, right=tmp, how='left', on='target')
        output = output.set_index('target').stack().reset_index()
        output.columns = ['target','test_name','score']
        sns.barplot(data=output, y='target', hue='test_name', x='score')
        return output
