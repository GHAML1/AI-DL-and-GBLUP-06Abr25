import numpy as np; import pandas as pd; 
from sklearn.preprocessing import StandardScaler; from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold; from scipy.stats import pearsonr; 
from bayes_opt import BayesianOptimization, UtilityFunction #https://github.com/fmfn/BayesianOptimization #!pip install bayesian-optimization
import pyreadr, sys, datetime, time, os, glob, random; from patsy import dmatrix
import matplotlib.pyplot as plt; from statsmodels.graphics.api import abline_plot
import torch as tch
print("Mensaje", flush=True)  # Opción 1
GPU = True if tch.cuda.is_available() else False
#Server, GPU =  True, False
#if GPU==False:
Num_Threads = 1
os.environ["OMP_NUM_THREADS"] = str(Num_Threads)
os.environ["MKL_NUM_THREADS"] = str(Num_Threads)
tch.set_num_threads(Num_Threads)
tch.set_num_interop_threads(Num_Threads)  # Limitar los hilos entre operaciones
Dir = os.getcwd()
Dir_Progs = Dir 
Dir_Progs = os.path.normpath(Dir_Progs)# sys.path is a list of absolute path strings
sys.path.append(Dir_Progs)
from Funs_Auxs import *
Match_f
from MLP_v8_0 import *
Predictor = 'GID'   
#Datos
os.getcwd()
Dir_DataSet =  os.path.normpath(os.path.join(Dir,'/dat-PP/'))
Folds_df = pd.read_csv(os.path.join(Dir_DataSet,'10Folds_df.csv'))
Folds_df.head()
DataSets =  ['Wheat_'+str(j+1)+'_AL' for j in range(6)]
DataSets
for dn in range(1,6+1):  # To run the deep learning models on the first 6 datasets
    #dn = 2
    DataNumber = dn-1
    Dir_dat = os.path.join(Dir_DataSet,DataSets[DataNumber]+'.RData')
    Dir_dat = ''.join(Dir_dat)
    OD = pyreadr.read_r(Dir_dat) 
    print(OD.keys())
    Pheno =  OD['Pheno']
    Pheno = Pheno.rename(columns={'Line':'GID'})
    Pheno.head()
    list(Pheno.columns)
    Traits = ["GY"]#,'T2','T3','T4' ]#colnames(Pheno)[4:7]
    for t in range(1,2):  
        print('DataSet',DataSets[DataNumber])
        #Response
        y = Pheno[Traits[t-1]]
        Pos_NA = np.where(y.isna())
        y =  y.to_numpy(copy=True)
        y = y.astype(float)
        y = np.reshape(y,(len(y),1))    
   
        #Markers
        XM = OD['Markers']
        np.shape(XM)
        print(XM.iloc[:,0:5].head(5))
        #XM.index = pd.RangeIndex(0,np.shape(XM)[0],1)
        XM =  XM.apply(np.round)
        XM.nunique().describe()
        np.unique(np.unique(XM,axis=1))
        print('Marker Data',np.shape(XM))
        #XM = tf.keras.utils.to_categorical(y=XM,num_classes=2)
        #np.shape(XM)#Para CNN
        Scaler_Markers = StandardScaler()#with_mean=False)
        XM = Scaler_Markers.fit_transform(XM)#xL_tr
        #print(Scaler_Markers.mean_)
        np.min(np.sqrt(Scaler_Markers.var_))
    
        G = np.matmul(XM,np.transpose(XM))/np.shape(XM)[1]
        G = pd.DataFrame(G)
        G5  =  XR2_f(G)
        np.shape(G5)
        Order = OD['Markers'].index # Define el orden deseado de las categorías
        #Convierte la columna 'GID' en una categoría ordenada
        Pheno['GID'] = pd.Categorical(Pheno['GID'], categories=Order, ordered=True)
        ZL = pd.get_dummies(Pheno['GID'],columns=['GID'])
        np.mean(ZL.columns==OD['Markers'].index)
        XL = np.matmul(ZL,G5[Match_f(ZL.columns,OD['Markers'].index),:])#
        XL = XL.to_numpy()
        ZL =  None   
        GIDs = Pheno['GID'].unique()
        Fold_df =  Folds_df[Folds_df['DataSet']==DataSets[DataNumber]]
        Fold_df.head()
        K = len(Fold_df['Fold'].unique())        
        x_ls_all = [XL]#
        Params = {}
        Hypers_Fixed = {'Epochs': 128,'Batch_size':len(y),'Seed':42,'Iters':250,
                        'IK': 10,'Use':10} 
        Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Definir el dispositivo (GPU si está disponible, de lo contrario CPU)
        Hypers_Fixed['Device'] =  Device#DataNumber = dn-1
        Dir_Outs0 = os.path.join(Dir,'Outs-Mod-v8_0-Shared'+'-Iters-'+str(Hypers_Fixed['Iters']),DataSets[dn-1])#+str(Hypers_Fixed['IK'])+'Use'+str(Hypers_Fixed['Use']))#+'-Trait'+Traits[t-1])+'-Iters-'+str(Hypers_Fixed['Iters']))
        if not os.path.exists(Dir_Outs0):
            os.makedirs(Dir_Outs0, exist_ok=True)#mkdir(Dir_Outs0)
        else:
            print(f"El directorio '{Dir_Outs0}' ya existe.")
        for k in range(1,K+1):
            Fold = np.repeat(100,len(y))
            Fold[Fold_df['Pos_tst'][Fold_df['Fold']==k].to_numpy()-1] = k#
            np.mean(Fold==k)
            x_ls_tr, y_tr = [x_ls_all[m][Fold!=k,:] for m in range(len(x_ls_all))], y[Fold!=k]
            Scaler_y = StandardScaler()
            y_tr = Scaler_y.fit_transform(y_tr)
            dat_tr = {'x_ls_tr':x_ls_tr,'y_tr':y_tr}        
            Hypers_Fixed['IFold'] = SIFold(dat_tr['y_tr'], K= Hypers_Fixed['IK'], nq=10, random_state=Hypers_Fixed['Seed'])   
            Time = time.time()
            def f_O(**kwargs):#nHL1,Units1,nHL2,Units2,nHLB2,UnitsB2,ll,llr):# nHL3,Units3 and (2**nHL3<=Units3)
                nHL1, Units1 = kwargs['nHL1'],kwargs['Units1']#,kwargs['nHL2'] ,kwargs['Units2']  
                #nHLB2, UnitsB2 = kwargs['nHLB2'],kwargs['UnitsB2']
                ll,llr, lwd = kwargs['ll'], kwargs['llr'],kwargs['lwd']#,kwargs['llo']]
                Ind = np.all(2**nHL1<=Units1)# and 2**nHL2<=Units2 and 2**nHLB2<=UnitsB2)
                if Ind:
                    nHLs, Units = [nHL1], [Units1]#nHL3
                    nHL_ls, Units = [int(nHLs[j]) for j in range(len(nHLs))], [int(Units[j]) for j in range(len(nHLs))]
                    Units_ls = [[ int(Units[m]/(2**j)) for j in range(nHL_ls[m]) ]   for m in range(len(x_ls_all))]
                    #nHLB2 =  int(1)
                    #UnitsB2_ls = []# int(UnitsB2/(2**j)) for j in range(nHLB2)]            
                    Hypers_R = {'UnitsB1_lsls':Units_ls,#'UnitsB2_ls':UnitsB2_ls,
                                'l':np.exp(ll),'lr':np.exp(llr),'wd':np.exp(lwd)}
                    Hypers_p = {**Hypers_Fixed,**Hypers_R}#{**Hypers_a,**{'wd':np.exp(lwd),'Pat':int(Pat)}}
                    #print(Hypers_p) Hypers_Fixed
                    Val_metric = IKFCV_f(dat_tr,Hypers_p,IK = Hypers_Fixed['IK'],
                                         Use = Hypers_Fixed['Use'],PlotTraining=False)
                    MSEP_Val = np.mean(Val_metric['Val_metric'])#-sigma**2
                    if np.isnan(MSEP_Val) or np.isinf(MSEP_Val) or MSEP_Val> np.var(y_tr)*1000:
                        MSEP_Val = np.var(y_tr)*1000#-1e5
                    return -MSEP_Val
                else:
                    #print('Units1',Units1,'nHL1',nHL1)
                    return -np.var(y_tr)*1000#-1e5
            Bounds = {'nHL1':[1,4],#'nHL2':[1,5],# 'nHL3':[1,4],
                      'Units1':[64,512],#'Units2':[32,1032],# 'Units2':(8,128),
                      #'nHLB2':[1,3], #'UnitsB2':[8,128],
                     'll':[np.log(np.e**- 10),np.log(np.e**-1)],
                     'llr':[np.log(np.e**-10),np.log(np.e**-0)],#,'Pat':(1,128)}
                     'lwd':[np.log(1-0.95),np.log(1-0.05)]}#Entre 5 y 95% se reduce el lr,'ll2':[np.log(1e-6),np.log(1e-1)],
                     #'llo':[np.log(1e-6),np.log(1e-1)],'llr':[np.log(1e-6),np.log(1e-1)],
                     
            BO = BayesianOptimization(f=f_O,pbounds=Bounds,random_state=50,verbose=1)
            #xi=0.01 # Mayor explotación (default), xi=0.1# Balance entre exploración y explotación
            #xi=1.0# Alta exploración, xi=2.0# Exploración muy alta
            #kappa=0.01#Baja exploración, kappa=2.576)  # Balance entre exploración y explotación (default)
            #kappa=10 #alta exploración,  kappa=20  # Exploración muy alta
            #Utility = UtilityFunction(kind="pi", xi=0.01)  # Probability of Improvement (PI), xi controla la exploración
            xi = 0.1#0.01, 1
            Utility = UtilityFunction(kind="ei", xi=xi)  # Expected Improvement (EI), valores más altos de xi fomentan la exploración
            #Utility = UtilityFunction(kind="ucb", kappa=2.576)  # Upper Confidence Bound (UCB), kappa controla la exploración
            BO.maximize(init_points=5, n_iter=Hypers_Fixed['Iters'],  acquisition_function = Utility)# Por defecto usa Expected Improvement (EI)
            #BO.maximize(init_points=5, n_iter=Hypers_Fixed['Iters'])# Por defecto usa Expected Improvement (EI)
            print(BO.max)
            #BO.set_bounds(new_bounds=Bounds)
            #BO.maximize(init_points=1, n_iter=Iters)
            #Optimal parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            Pars_O = BO.max['params']
            #print(Pars_O)
            nHLs, Units, l =  [], [], []
            for i in range(len(x_ls_all)):#Cambiar 6 por nI+1 para cuando son nI inputs
                nHLs.append(int(Pars_O[''.join(['nHL',str(i+1)])]))
                Units.append(int(Pars_O[''.join(['Units',str(i+1)])]))
                #l.append(np.exp(Pars_O['ll'+str(i+1)]))
            l = np.exp(Pars_O['ll'])
            nHL_ls, Units = [int(nHLs[j]) for j in range(len(nHLs))], [int(Units[j]) for j in range(len(nHLs))]
            Units_ls = [[ int(Units[m]/(2**j)) for j in range(nHL_ls[m]) ]   for m in range(len(x_ls_all))]
            #nHLB2 =  int(Pars_O['nHLB2'])
            #UnitsB2_ls = []#[int(Pars_O['UnitsB2']/(2**j)) for j in range(nHLB2)]
            Hypers_R_O = {'UnitsB1_lsls':Units_ls,#'UnitsB2_ls':UnitsB2_ls,
                        'l':l,'lr':np.exp(Pars_O['llr']),'wd':np.exp(Pars_O['lwd'])}
            Params_O = {**Hypers_Fixed,**Hypers_R_O}#{**Hypers_a,**{'wd':np.exp(lwd),'Pat':int(Pat)}}
            Params_O
            
            #Prediction of tst >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            yp_tst  =  np.zeros((np.sum(Fold==k),1))#0*y_tst
            y_tst, x_ls_tst = y[Fold==k], [x_ls_all[m][Fold==k,:] for m in range(len(x_ls_all))]
            x_ls_tst = [torch.from_numpy(arr) for arr in x_ls_tst]
            x_ls_tst =  [x_ls_tst[m].to(torch.float32) for m in range(len(x_ls_all))]
            if torch.cuda.is_available():
                x_ls_tst = [tensor.to(Params_O['Device']) for tensor in x_ls_tst]
            for ik in range(1,Hypers_Fixed['Use']+1):
                DModel_O = MMMLP(Params_O, dat_tr)# Definir el modelo                
                Pos_itr = Hypers_Fixed['IFold'] != ik #Notar que es un arreglo. Pos_itr = [IFold[j] != ik for j in range(len(IFold))]
                #print(Pos_itr)
                DModel_O.fit(Pos_itr=Pos_itr, PlotTraining=False) # Entrenar el modelo                
                yp_tst_ik =  DModel_O.Model(x_ls_tst)#Scaler_y.inverse_transform(FM_O_ls['Model'].predict(x_ls_tst,verbose=0))# flatten()
                if torch.cuda.is_available():
                    yp_tst_ik = yp_tst_ik.cpu()
                yp_tst = yp_tst +  Scaler_y.inverse_transform(yp_tst_ik.detach().numpy())
            yp_tst = yp_tst/Hypers_Fixed['Use']
            # DModel_O = MMMLP(Params_O, dat)# Definir el modelo        
            # DModel_O.fit(Pos_itr=None, PlotTraining=True)# Entrenar el modelo        
            # yp_tst = DModel_O.Model(x_ls_tst)# Entrenar el modelo 
            # yp_tst =  yp_tst.cpu().detach().numpy()#[:,0] 
            #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            ##plt.scatter(y_tst,yp_tst,c='blue',s=5)
            #ax.scatter(y_tst,yp_tst,c='blue',s=5)
            #abline_plot(intercept=0, slope=1, ax=ax)
            #plt.show()
            Time = time.time()-Time
            np.mean((y_tst-yp_tst)**2)
            #Pos_Top tst
            #Matching_Top = np.mean(yp_tst>np.quantile(y[Fold!=k],0.80))
            PosTop_tst, PosTop_tstp = np.where(y_tst>np.quantile(y_tst,0.8))[0], np.where(yp_tst>np.quantile(yp_tst,0.8))[0]
            GIDTop_tst = GIDs[Fold==k][PosTop_tst]
            # #Matching_Top = np.mean([GIDs[Fold==k][i] in GIDTop_tst for i in range(np.sum(Fold==k))])# No tiene sentido
            Id_df = {'Iters':Hypers_Fixed['Iters'],
                     'Batch_size':Hypers_Fixed['Batch_size'],'GPU':str(GPU),
                     'AF':'EI','Xi':xi,'Folds':K,'IKUsed':str(Hypers_Fixed['IK'])+'Used'+str(Hypers_Fixed['Use']),
                     'DataSet':DataSets[DataNumber], 'Trait':Traits[t-1]}
            Matching_Top = np.mean([GIDs[Fold==k][PosTop_tstp][i] in GIDTop_tst for i in range(len(PosTop_tstp))])
            Tab_k = pd.DataFrame({**Id_df,**{'Fold':k,'Env':'NA', 'Cor':[np.corrcoef(y_tst[:,0],yp_tst[:,0])[0,1]],
                                  'MSE':[np.mean((y_tst-yp_tst)**2)],
                                  'NRMSE':[np.sqrt(np.mean((y_tst-yp_tst)**2))/np.mean(y_tst)],'Time':Time/3600,
                                  'Matching':Matching_Top}})
            
            #Save Preds                             
            Dir_Outs =  os.path.join(Dir_Outs0,'Preds-'+Predictor+'-Trait-'+str(Traits[t-1])+'.csv')
            df_Preds = pd.DataFrame({**Id_df,**{'Env':'NA',#dat_F.loc[np.where(Fold==k)]['Env'],
                                      'Fold':k,'y':y_tst[:,0],'yp':yp_tst[:,0]}})
            df_Preds.reset_index(inplace=True)
            #Tab_k2 = Metrics_df(df_Preds,Grp='Env')
            #Tab_k2.insert(0, 'Fold', k)
            if not os.path.exists(Dir_Outs):            
                df_Preds.to_csv(Dir_Outs,index=False)
            else:
                Existing_file =  pd.read_csv(Dir_Outs)
                df_Preds = pd.concat([Existing_file,df_Preds], ignore_index=True)
                df_Preds.to_csv(Dir_Outs,index=False)        
            #Save Tb-Smm
            Dir_Outs =  os.path.join(Dir_Outs0,'Tab-Smm-'+Predictor+'-Trait-'+str(Traits[t-1])+'.csv')    
            #Tab_k =  pd.concat([Tab_k,Tab_k2])
            if not os.path.exists(Dir_Outs):                        
                Tab_k.to_csv(Dir_Outs,index=False)
            else:
                Existing_file =  pd.read_csv(Dir_Outs)
                Tab_k = pd.concat([Existing_file,Tab_k], ignore_index=True)
                Tab_k.to_csv(Dir_Outs,index=False)       
            #Save Hypers
            Dir_Outs =  os.path.join(Dir_Outs0,'Hypers-'+Predictor+'-Trait-'+str(Traits[t-1])+'.csv')
            Hyper_df_k = pd.DataFrame(BO.max)
            Hyper_df_k['Parameter'], Hyper_df_k['Fold'] = Hyper_df_k.index, k
            Hyper_df_k
            if not os.path.exists(Dir_Outs):
                Hyper_df_k.to_csv(Dir_Outs,index=False)
            else:
                Hyper_df =  pd.read_csv(Dir_Outs)
                Hyper_df_k = pd.concat([Hyper_df,Hyper_df_k], ignore_index=True)
                Hyper_df_k.to_csv(Dir_Outs,index=False)   
            
            print('Fold',k)
        
    
