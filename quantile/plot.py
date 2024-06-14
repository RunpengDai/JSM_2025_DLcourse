import matplotlib.pyplot as plt
import torch

def plot(quants, data_train, preds, taus):
    x_test=torch.linspace(0,1,1000).unsqueeze(1)
    colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:cyan']
    methods=['NQ-Net','DQR','DQR*','NCQR']
    names=(r'$\tau=0.05$',r'$\tau=0.25$',r'$\tau=0.5$',r'$\tau=0.75$',r'$\tau=0.95$','Data')

    figs, axs = plt.subplots(1,4,figsize=(80,18))
    #ticksize=15;titlesize=32;llw=3;dlw=3;
    for m,method in enumerate(methods):
        axs[m].tick_params(axis='both', which='major', labelsize=35)
        axs[m].set_title('%s'% (methods[m]),fontdict={'family':'Times New Roman','size':75})
        axs[m].set_xlabel(r'$X$', fontdict={'family': 'Times New Roman', 'size': 35})
        axs[m].set_ylabel(r'$Y$', fontdict={'family': 'Times New Roman', 'size': 35})
        axs[m].set_xlim(0, 1)
        axs[m].set_ylim([-4,8])
        axs[m].plot(x_test, quants, alpha=0.9,lw=4)
        axs[m].scatter(data_train[:][0].data.numpy(), data_train[:][1].data.numpy(), color = "k", alpha=0.25,label='Data',s=30)
        axs[m].legend(names,loc='upper left',fontsize=35)
        for j in range(len(taus)):
            axs[m].plot(x_test, preds[m][:,j], color=colors[j],linestyle='--',alpha=0.9,lw=5)