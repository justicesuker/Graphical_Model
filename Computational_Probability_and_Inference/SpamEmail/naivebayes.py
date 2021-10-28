# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 13:55:26 2016

@author: suker

Important! This py script needs to specify the email folder in the order of testing, spam, ham.
To pass argv to a script in Spyder, go to the menu entry

Run > Configuration per file

then look for the option called

Command line options

on the dialog that appears after that, and finally enter the command line arguments you want to pass to the script.
"""

import sys
import os.path
import numpy as np
import collections

import util

USAGE = "%s <test data folder> <spam folder> <ham folder>"

def get_counts(file_list):
    """
    Computes counts for each word that occurs in the files in file_list.

    Inputs
    ------
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the number of files the
    key occurred in.
    """
    ### TODO: Comment out the following line and write your code here
#    raise NotImplementedError
    counts={}    
    cut_word_list={}
    n=0
    for i in file_list:
        cut_word_list[i]={}
        n=0
        for k in util.get_words_in_file(i):
            if k not in cut_word_list[i]:
                cut_word_list[i][n]=k
                n+=1
    for i in file_list:
        for k in cut_word_list[i]:
            if cut_word_list[i][k] in counts:
                counts[cut_word_list[i][k]]+=1
            else: 
                counts[cut_word_list[i][k]]=1   
    return counts

def get_log_probabilities(file_list):
    """
    Computes log-frequencies for each word that occurs in the files in 
    file_list.

    Input
    -----
    file_list : a list of filenames, suitable for use with open() or 
                util.get_words_in_file()

    Output
    ------
    A dict whose keys are words, and whose values are the log of the smoothed
    estimate of the fraction of files the key occurred in.

    Hint
    ----
    The data structure util.DefaultDict will be useful to you here, as will the
    get_counts() helper above.
    """
    ### TODO: Comment out the following line and write your code here
#    raise NotImplementedError
    counts = get_counts(file_list)
    l=len(file_list)
    log_pro=collections.defaultdict(lambda:-np.log(l+2))
    for k in counts:
        log_pro[k]=np.log((counts[k]+1)/(l+2))
    return log_pro
    
def learn_distributions(file_lists_by_category):
    """
    Input
    -----
    A two-element list. The first element is a list of spam files, 
    and the second element is a list of ham (non-spam) files.

    Output
    ------
    (log_probabilities_by_category, log_prior)

    log_probabilities_by_category : A list whose first element is a smoothed
                                    estimate for log P(y=w_j|c=spam) (as a dict,
                                    just as in get_log_probabilities above), and
                                    whose second element is the same for c=ham.

    log_prior_by_category : A list of estimates for the log-probabilities for
                            each class:
                            [est. for log P(c=spam), est. for log P(c=ham)]
    """
    ### TODO: Comment out the following line and write your code here
    sum=0    
    log_probabilities_by_category={}
    log_prior_by_category=[]
    prior_by_category={}
    n=0
    for i in file_lists_by_category:
        log_probabilities_by_category[n]=get_log_probabilities(i)
        prior_by_category[n]=len(i)
        sum=sum+prior_by_category[n]
        n=n+1
    for i in range(2):
        log_prior_by_category.append(np.log(prior_by_category[i]/sum))
    return log_probabilities_by_category, log_prior_by_category 
#    raise NotImplementedError
    

def classify_email(email_filename,
                   log_probabilities_by_category,
                   log_prior_by_category):
    """
    Uses Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    email_filename : name of the file containing the email to be classified

    log_probabilities_by_category : See output of learn_distributions

    log_prior_by_category : See output of learn_distributions

    Output
    ------
    One of the labels in names.
    """
    ### TODO: Comment out the following line and write your code here
    words=util.get_words_in_file(email_filename)
    y0={}
    y1={}
    result=[0,0]
    l1=len(log_probabilities_by_category[0])
    l2=len(log_probabilities_by_category[1])
    #print(log_probabilities_by_category[0])
    for j in words:
        if j in log_probabilities_by_category[0]:
            y0[j]=1
        else:
            y0[j]=0
    for j in words:
        if j in log_probabilities_by_category[1]:
            y1[j]=1
        else:
            y1[j]=0            
    for j in words:
        if y0[j]==0:
            result[0]+=-np.log(l1+2)
        if y0[j]==1:
            result[0]+=log_probabilities_by_category[0][j]
    for j in words:
        if y1[j]==0:
            result[1]+=-np.log(l2+2)
        if y1[j]==1:
            result[1]+=log_probabilities_by_category[1][j]
    for i in range(2):
        result[i]=result[i]+log_prior_by_category[i]
    if result[0]>=result[1]:
        return 'spam'
    else:
        return 'ham'

def classify_emails(spam_files, ham_files, test_files):
    # DO NOT MODIFY -- used by the autograder
    log_probabilities_by_category, log_prior = \
        learn_distributions([spam_files, ham_files])
    estimated_labels = []
    for test_file in test_files:
        estimated_label = \
            classify_email(test_file, log_probabilities_by_category, log_prior)
        estimated_labels.append(estimated_label)
    return estimated_labels

def main():
    ### Read arguments
    if len(sys.argv) != 4:
        print(USAGE % sys.argv[0])
    testing_folder = sys.argv[1]
    (spam_folder, ham_folder) = sys.argv[2:4]

    ### Learn the distributions
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
    (log_probabilities_by_category, log_priors_by_category) = \
            learn_distributions(file_lists)

    # Here, columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    performance_measures = np.zeros([2,2])

    ### Classify and measure performance
    for filename in (util.get_files_in_folder(testing_folder)):
        ## Classify
        label = classify_email(filename,
                               log_probabilities_by_category,
                               log_priors_by_category)
        ## Measure performance
        # Use the filename to determine the true label
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[true_index, guessed_index] += 1


        # Uncomment this line to see which files your classifier
        # gets right/wrong:
        #print("%s : %s" %(label, filename))

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],
                      totals[0],
                      correct[1],
                      totals[1]))

if __name__ == '__main__':
    main()
