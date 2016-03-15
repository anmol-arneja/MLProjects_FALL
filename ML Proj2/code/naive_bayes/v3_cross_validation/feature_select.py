'''
COMP-598 - Applied Machine Learning 
Project 2 - Classification

feature_select.py = This file performs feature selection for a given data
@authors: Sandra Maria Nawar
          Timardeep Kaur
          Daniel Galeano
October 20th, 2015
'''

#*******************************************************************************
# IMPORT LIBRARIES AND TOOLS
#*******************************************************************************
import logging
import numpy as np
import defines
import heapq
#*******************************************************************************
#CLASS DEFINITIONS
#*******************************************************************************
class common_data:
    def __init__(self, x, y, Nx, Ny, x_list, y_list):
        self.x = x                  # Training Data
        self.y = y                  # Test Data
        self.N  = len(x)            # Number of string samples
        self.Nx = Nx                # Array with number of strings with features/words
        self.Nxn = len(x_list) - Nx # Array with number of strings without features/words
        self.Ny = Ny                # Array with number of strings with labels/topics
        self.Nyn = len(y_list) - Ny # Array with number of strings without labels/topics
        self.x_list = x_list        # List of all features/words
        self.y_list = y_list        # List of all labels/topics


class MutualInfo:
    def __init__(self, label_id, feat_num):
        self.id  = label_id                 # Label ID

        # Initialize Counters
        self.Nx_y   = np.zeros(feat_num)  # Number of samples with feature and label
        self.Nx_yn  = np.zeros(feat_num)  # Number of samples with feature and without label
        self.Nxn_y  = np.zeros(feat_num)  # Number of samples without feature and with label
        self.Nxn_yn = np.zeros(feat_num)  # Number of samples wihout feature and without label
        self.mi     = np.zeros(feat_num)    # Matrix with MI values


#*******************************************************************************
# FUNCTION DEFINITIONS
#*******************************************************************************

def print_features(MI, COMMON_DATA, LABEL_STR):
     # Pick features with highest mutual information values
    max_mi = heapq.nlargest(defines.FEATURE_SELECTED, range(len(MI.mi)), MI.mi.take)
    print ('%s'%(LABEL_STR))
    logging.info('=========  %s  ============'%(LABEL_STR))
    for i in max_mi:
        print('[%s]' %(COMMON_DATA.x_list[i]))
        logging.info('[%s]'  %(COMMON_DATA.x_list[i]))

#*******************************************************************************
# Get selected features based on mutual information
def get_selected_features(CSV_DATA):
    logging.basicConfig(filename='feat_select.log',level=logging.DEBUG)
    logging.info('get_selected_features: ENTER')

    # Collect common data needed for all calculations
    COMMON_DATA = get_common_data(CSV_DATA)
    logging.info('get_selected_features: Got common data')

    label_num = len(COMMON_DATA.y_list) # Number of labels
    feat_num  = len(COMMON_DATA.x_list) # Number of features

    # Create mutual information instance for each label
    AUTHOR_MI       = get_mutual_info(defines.LABEL_AUTHOR , COMMON_DATA)
    print_features(AUTHOR_MI, COMMON_DATA, 'AUTHOR')

    MOVIES_MI       = get_mutual_info(defines.LABEL_MOVIES, COMMON_DATA)
    print_features(MOVIES_MI, COMMON_DATA, 'MOVIES')

    MUSIC_MI        = get_mutual_info(defines.LABEL_MUSIC, COMMON_DATA)
    print_features(MUSIC_MI, COMMON_DATA, 'MUSIC')

    INTERVIEW_MI    = get_mutual_info(defines.LABEL_INTERVIEW, COMMON_DATA)
    print_features(INTERVIEW_MI, COMMON_DATA, 'INTERVIEW')

    print('Finished selecting features!')
    logging.info('=========  done!  ============')
    
    
#*******************************************************************************
# Get mutual information for a given label
def get_mutual_info(LABEL, COMMON_DATA):
    logging.info('get_mutual_info: ENTER with label %d' %(LABEL))
    print('get_mutual_info: ENTER with label %d' %(LABEL))

    
    feat_num = len(COMMON_DATA.x_list)

    # Create calss instance for a given label
    LABEL_CLASS = MutualInfo(LABEL, feat_num)

    # Find Counters
    print('Find counters')
    for index in range(len(COMMON_DATA.x)):
        u_str = np.unique(COMMON_DATA.x[index])
        if index % 1000 == 0:
            print('[%d]'%(index))
        if int(COMMON_DATA.y[index]) == LABEL:
            match = np.array([ len(u_str[xval == u_str]) for xval in COMMON_DATA.x_list ])
            LABEL_CLASS.Nx_y += match
            LABEL_CLASS.Nxn_y += 1 - match
        else:
            match = np.array([ len(u_str[xval == u_str]) for xval in COMMON_DATA.x_list ])
            LABEL_CLASS.Nx_yn += match
            LABEL_CLASS.Nxn_yn += 1 - match

    # Initialize partial counters
    print('\n\nFind partial counters')
    U1_C1 = np.zeros(feat_num)
    U0_C1 = np.zeros(feat_num)
    U1_C0 = np.zeros(feat_num)
    U0_C0 = np.zeros(feat_num)

    # Calculate Mutual Information
    # U = 1 and C = 1
    Nxy_d_N = LABEL_CLASS.Nx_y / COMMON_DATA.N      # Nxy / N
    Nxy_t_N = LABEL_CLASS.Nx_y * COMMON_DATA.N      # Nxy * N
    Nx_t_Ny =  COMMON_DATA.Nx * COMMON_DATA.Ny[LABEL]     # Nx * Ny
    LOG_RATIO = np.zeros(feat_num)
    LOG_RATIO[Nx_t_Ny > 0] = Nxy_t_N[Nx_t_Ny > 0] / Nx_t_Ny[Nx_t_Ny > 0]    
    if max(LOG_RATIO) > 0:
        LOG_RATIO[LOG_RATIO > 0] = np.log2(LOG_RATIO[LOG_RATIO > 0])
        U1_C1 = Nxy_d_N * LOG_RATIO
    

    # U = 0 and C = 1
    Nxy_d_N = LABEL_CLASS.Nxn_y / COMMON_DATA.N     
    Nxy_t_N = LABEL_CLASS.Nxn_y * COMMON_DATA.N     
    Nx_t_Ny =  COMMON_DATA.Nxn * COMMON_DATA.Ny[LABEL]    
    LOG_RATIO = np.zeros(feat_num)
    LOG_RATIO[Nx_t_Ny > 0] = Nxy_t_N[Nx_t_Ny > 0] / Nx_t_Ny[Nx_t_Ny > 0]    
    if max(LOG_RATIO) > 0:
        LOG_RATIO[LOG_RATIO > 0] = np.log2(LOG_RATIO[LOG_RATIO > 0])
        U0_C1 = Nxy_d_N * LOG_RATIO

    # U = 1 and C = 0
    Nxy_d_N = LABEL_CLASS.Nx_yn / COMMON_DATA.N    
    Nxy_t_N = LABEL_CLASS.Nx_yn * COMMON_DATA.N      
    Nx_t_Ny =  COMMON_DATA.Nx * COMMON_DATA.Nyn[LABEL]  
    LOG_RATIO = np.zeros(feat_num)
    LOG_RATIO[Nx_t_Ny > 0] = Nxy_t_N[Nx_t_Ny > 0] / Nx_t_Ny[Nx_t_Ny > 0]    
    if max(LOG_RATIO) > 0:
        LOG_RATIO[LOG_RATIO > 0] = np.log2(LOG_RATIO[LOG_RATIO > 0])
        U1_C0 = Nxy_d_N * LOG_RATIO

    # U = 0 and C = 0
    Nxy_d_N = LABEL_CLASS.Nxn_yn / COMMON_DATA.N     
    Nxy_t_N = LABEL_CLASS.Nxn_yn * COMMON_DATA.N   
    Nx_t_Ny =  COMMON_DATA.Nxn *  COMMON_DATA.Nyn[LABEL]
    LOG_RATIO = np.zeros(feat_num)
    LOG_RATIO[Nx_t_Ny > 0] = Nxy_t_N[Nx_t_Ny > 0] / Nx_t_Ny[Nx_t_Ny > 0]     
    if max(LOG_RATIO) > 0:
        LOG_RATIO[LOG_RATIO > 0] = np.log2(LOG_RATIO[LOG_RATIO > 0])
        U0_C0 = Nxy_d_N * LOG_RATIO

    LABEL_CLASS.mi = U1_C1 + U0_C1 +  U1_C0 + U0_C0

    return LABEL_CLASS


#*******************************************************************************
# Get common data from CSV data
def get_common_data(CSV_DATA):
    # Extract array of strings from CSV data
    x_str = CSV_DATA[:,0]

    # Strip puntuation and other useless characters from each string
    print('strip useless characters')
    # Make all string lower case
    x_str = np.array([str_sample.lower()   for str_sample in x_str])
    x_str = np.array([str_sample.replace(',','').replace('"',' ')
                        .replace('.',' ').replace('?',' ').replace('-',' ').replace('\'t ',' ')
                        .replace(':',' ').replace('\'re','').replace('\'m',' ')
                        .replace(';',' ').replace('`',' ').replace('__eos__',' ')
                        .replace('\'s',' ').replace('"',' ').replace('%',' ')
                        .replace('\'ve',' ').replace('some',' ').replace('let',' ')
                        .replace(' is ',' ').replace(' was ',' ').replace(' the ',' ')
                        .replace(' that ',' ').replace(' and ',' ').replace(' you ',' ')
                        .replace(' i ',' ').replace(' for ',' ').replace(' a ',' ')
                        .replace(' your ',' ').replace(' of ',' ').replace(' it ',' ')
                        .replace(' to ',' ').replace(' in ',' ').replace(' this ',' ')
                        .replace(' so ',' ').replace(' what ',' ').replace('?',' ')
                        .replace('where',' ').replace(' how ',' ').replace(' when ',' ')
                        .replace(' my ',' ').replace(' but ',' ').replace('have',' ')
                        .replace('know',' ').replace(' just ',' ').replace(' we ',' ')
                        .replace(' he ',' ').replace(' she ',' ').replace('they',' ')
                        .replace('with',' ').replace(' think ',' ').replace(' be ',' ')
                        .replace(' out ',' ').replace(' well ',' ').replace(' about ',' ')
                        .replace(' all ',' ').replace(' like ',' ').replace(' very ',' ')
                        .replace(' not ',' ').replace(' no ',' ').replace(' on ','')
                        .replace(' yeah ',' ').replace('because',' ').replace(' there ',' ')
                        .replace(' do ',' ').replace(' or ',' ').replace(' from ',' ')
                        .replace(' at ',' ').replace(' were ',' ').replace('who',' ')
                        .replace(' as ',' ').replace(' one ',' ').replace('would',' ')
                        .replace(' me ',' ').replace(' are ',' ').replace(' had ',' ')
                        .replace(' an ',' ').replace(' right ',' ').replace(' if ',' ')
                        .replace(' can ',' ').replace(' us ',' ').replace('don\'t',' ')
                        .replace(' kind ',' ').replace(' much ',' ').replace(' mean ',' ')
                        .replace(' really ',' ').replace(' up ',' ').replace('his',' ')
                        .replace(' if ',' ').replace(' then ',' ').replace(' oh ',' ')
                        .replace(' lot ',' ').replace(' them ',' ').replace(' get ',' ')
                        .replace(' go ',' ').replace(' bye ',' ').replace(' some ',' ')
                        .replace(' let ',' ').replace(' yes ',' ').replace(' by ',' ') 
                        .replace(' their ',' ').replace(' did ',' ').replace(' which ',' ')
                        .replace(' good ',' ').replace(' being ',' ').replace(' here ',' ') 
                        .replace(' these ',' ').replace(' most ',' ').replace(' could ',' ') 
                        .replace(' something ',' ').replace(' new ',' ').replace(' her ',' ') 
                        .replace(' same ',' ').replace(' took ',' ').replace(' two ',' ') 
                        .replace(' also ',' ').replace(' other ',' ').replace(' getting ',' ') 
                        .replace(' thing ',' ').replace(' things ',' ').replace(' want ',' ')
                        .replace(' him ',' ').replace(' been ',' ').replace(' our ',' ') 
                        .replace(' only ',' ').replace(' didn\'t ',' ').replace('  ',' ') 
                        .replace(' has ',' ').replace(' going ',' ').replace(' our ',' ') 
                        .replace(' now ',' ').replace(' make ',' ').replace(' many ',' ') 
                        .replace(' after ',' ').replace(' done ',' ').replace(' always ',' ') 
                        .replace(' maybe ',' ').replace('  ',' ').replace('  ',' ')
                        .replace(' any ',' ').replace(' true ',' ').replace(' c ',' ') 
                        .replace(' way ',' ').replace(' guy ',' ').replace(' ok ',' ') 
                        .replace(' great ',' ').replace(' bit ',' ').replace(' than ',' ') 
                        .replace(' couldn\'t ',' ').replace(' must ',' ').replace(' first ',' ') for str_sample in x_str])

    

    # Tokenize each string into words
    print('tokenize each string')
    x = np.array([str_sample.split()  for str_sample in x_str])

    # Get labels from CSV data
    y = np.array(CSV_DATA[:,1])

    # Build list unique features
    print('strip useless words')
    x_value_list = [] 

    i = 1
    for str_sample in x:
        i += 1
        if (i % 500) == 0:
            print('[%d] L = %d || ' %( i, len(x_value_list)))

        for word_sample in str_sample:
            if not word_sample in x_value_list:
                x_value_list.append(word_sample);
               


    # Build list of unique labels
    y_value_list = np.unique(y)


    # Find Nx = Number of strings containing feature [word] 'x'
    print('\n\nFind Nx')
    Nx = np.zeros(len(x_value_list))
    i = 0
    for str_sample in x:
        i += 1
        if (i % 500) == 0:
            print('[%d]' %( i))
        unique_str_sample = np.unique(str_sample)
        if len(unique_str_sample) > 0:
            Nx += np.array([ len(unique_str_sample[xval == unique_str_sample]) for xval in x_value_list ])
        

    # Find Ny = Number of documents containing label [topic] 'y'
    print('\n\n Find Ny')
    Ny = np.array([ len(y[y==yval]) for yval in y_value_list ])

    COMMON_DATA = common_data(x, y, Nx, Ny, x_value_list, y_value_list)
    return COMMON_DATA
       