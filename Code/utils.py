import numpy as np
import matplotlib.pyplot as plt

def import_data(results):
    rewards = results.rewards
    actions = results.actions
    oms = results.oms
    mus = results.mus
    std = results.stds
    return rewards, actions, oms, mus, std

def get_outcome(rewards, actions):
    outcome = rewards + actions*2
    outcome[outcome<0] = 0
    return outcome


def analyse_few_games(results, string):
    rewards, actions, oms, mus, std = import_data(results)
    outcome = get_outcome(rewards, actions)

    est_prob = np.sum(outcome==2,axis=1)/len(outcome[0,:])
    check_prob = np.sum(outcome==4,axis=1)/len(outcome[0,:])
    tot_reward = np.sum(rewards,axis=1)
    
    #  Plot
    fig, axs = plt.subplots(6, 1, figsize=(12, 8), sharex=True)

    from matplotlib import colors
    #axs[0].pcolormesh(actions, cmap='binary')
    #axs[0].set_title('Actions')
    # use the cmap which has blue for negative and red for positive values

    #plot actions, color code them by action array with the code -1: "b", 1 "r", 2: "k", 4: "w" 
    cmap = colors.ListedColormap(['b','r','k',"lightgreen"])
    bounds = [-0.5,0.5,1.5,2.5,3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    colors = ["b","r","k","w"]
    d = axs[0].pcolormesh(outcome, cmap=cmap, norm=norm)
    axs[0].grid(True)
    cb = plt.colorbar(d, ax = axs[0], orientation="vertical")
    cb.set_label('Action')
    cb.set_ticks([])
    plt.tight_layout()
    axs[0].grid(True)

    #mu
    dmu = mus
    dmu_plot = axs[1].pcolormesh(dmu, cmap="Reds", vmin=0, vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[1], orientation="vertical")
    cb.set_label(r'Estimate $\mu$')
    plt.tight_layout()
    plt.grid()


    #Ustd
    sig_plot = axs[2].pcolormesh(std, cmap="Reds", vmin=0, vmax=np.max(std))

    cb = plt.colorbar(np.log10(sig_plot), ax = axs[2], orientation="vertical")
    cb.set_label('Uncertainty $\sigma$')
    plt.tight_layout()
    plt.grid()


    #real om
    dmu = oms 
    dmu_plot = axs[3].pcolormesh(dmu, cmap="bwr", vmin=-np.max(np.abs(dmu)), vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[3], orientation="vertical")
    cb.set_label('Real $\omega$')
    plt.tight_layout()
    plt.grid()


    #error
    dom = oms - mus
    om_plot = axs[4].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[4], orientation="vertical")
    cb.set_label('Estimation error')
    plt.tight_layout()
    axs[1].grid(True)


    #error
    dom = np.abs(oms) - np.abs(mus)
    om_plot = axs[5].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[5], orientation="vertical")
    cb.set_label('Estimation error abs')
    plt.tight_layout()
    plt.title(string)
    axs[1].grid(True)
    plt.savefig("figures/games"+str(string)+".png")

def analyse_few_games2(results, string):
    rewards, actions, oms, mus, std = import_data(results)
    rewards_to_plot = np.zeros(rewards.shape)
    times = np.zeros(rewards.shape)
    rewards_to_plot[rewards<0] = -1  #0 - success, #-1 -failure, times...
    rewards_to_plot[rewards==1] = 1
    times[rewards==0] = actions[1][rewards==0]
    maxtime = np.max(times)
    rewards_to_plot[rewards_to_plot==0] = None
    times[times==0] = None

    #est_prob = np.sum(outcome==2,axis=1)/len(outcome[0,:])
    #check_prob = np.sum(outcome==4,axis=1)/len(outcome[0,:])
    #tot_reward = np.sum(rewards,axis=1)
    
    #  Plot
    fig, axs = plt.subplots(6, 1, figsize=(12, 8), sharex=True)

    from matplotlib import colors
    #axs[0].pcolormesh(actions, cmap='binary')
    #axs[0].set_title('Actions')
    # use the cmap which has blue for negative and red for positive values

    #plot rewards with -1 -> blue, 0-> red, and >1 -> colormap
    
    cmap_rewards = "bwr"
    cmap_times = "binary"

    #colors = ["b","r","k","w"]
    d = axs[0].pcolormesh(rewards_to_plot, cmap=cmap_rewards)
    d2 = axs[0].pcolormesh(times, cmap=cmap_times,vmax = maxtime*2)
    axs[0].grid(True)
    cb = plt.colorbar(d2, ax = axs[0], orientation="vertical")
    cb.set_label('Action')
    cb.set_ticks([])
    plt.tight_layout()
    axs[0].grid(True)

    #mu
    dmu = mus
    dmu_plot = axs[1].pcolormesh(dmu, cmap="Reds", vmin=0, vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[1], orientation="vertical")
    cb.set_label('Estimate $\mu$')
    plt.tight_layout()
    plt.grid()


    #Ustd
    sig_plot = axs[2].pcolormesh(std, cmap="Reds", vmin=0, vmax=np.max(std))

    cb = plt.colorbar(sig_plot, ax = axs[2], orientation="vertical")
    cb.set_label('Uncertainty $\sigma$')
    plt.tight_layout()
    plt.grid()


    #real om
    dmu = oms 
    dmu_plot = axs[3].pcolormesh(dmu, cmap="bwr", vmin=-np.max(np.abs(dmu)), vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[3], orientation="vertical")
    cb.set_label(r'Real $\omega$')
    plt.tight_layout()
    plt.grid()


    #error
    dom = oms - mus
    om_plot = axs[4].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[4], orientation="vertical")
    cb.set_label('Estimation error')
    plt.tight_layout()
    axs[1].grid(True)


    #error
    dom = np.abs(oms) - np.abs(mus)
    om_plot = axs[5].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[5], orientation="vertical")
    cb.set_label('Estimation error abs')
    plt.tight_layout()
    plt.title(string)
    axs[1].grid(True)
    plt.savefig("figures/games"+str(string)+".png")

def analyse_few_games3(results, string):
    rewards, actions, oms, mus, std = import_data(results)
    rewards_to_plot = np.zeros(rewards.shape)
    times = np.zeros(rewards.shape)
    rewards_to_plot[rewards<0] = -1  #0 - success, #-1 -failure, times...
    rewards_to_plot[rewards==1] = 1
    times[rewards==0] = actions[1][rewards==0]
    maxtime = np.max(times)
    rewards_to_plot[rewards_to_plot==0] = None
    times[times==0] = None

    #est_prob = np.sum(outcome==2,axis=1)/len(outcome[0,:])
    #check_prob = np.sum(outcome==4,axis=1)/len(outcome[0,:])
    #tot_reward = np.sum(rewards,axis=1)
    std = std*mus
    #  Plot
    fig, axs = plt.subplots(6, 1, figsize=(6, 6), sharex=True)
    plt.subplots_adjust(hspace=0.0)
    from matplotlib import colors
    #axs[0].pcolormesh(actions, cmap='binary')
    #axs[0].set_title('Actions')
    # use the cmap which has blue for negative and red for positive values

    #plot rewards with -1 -> blue, 0-> red, and >1 -> colormap
    for k in range(6):
        axs[k].set_yticks([0,1,2,3])


    cmap_rewards = "bwr"
    cmap_times = "binary"

    #colors = ["b","r","k","w"]
    d = axs[0].pcolormesh(rewards_to_plot, cmap=cmap_rewards)
    d2 = axs[0].pcolormesh(times, cmap=cmap_times,vmax = maxtime*2)
    axs[0].grid(True)
    cb = plt.colorbar(d2, ax = axs[0], orientation="vertical")
    cb.set_label('Est. time')
    cb.set_ticks([])
    plt.tight_layout()
    axs[0].grid(True)
    axs[0].set_ylabel("Repetitions")
    #mu
    dmu = mus
    
    dmu_plot = axs[1].pcolormesh(dmu, cmap="Reds", vmin=0, vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[1], orientation="vertical")
    cb.set_label('Est. $\omega$')
    plt.tight_layout()
    axs[1].grid(True)
    axs[1].set_ylabel("Repetitions")


    #Ustd
    sig_plot = axs[2].pcolormesh(std, cmap="Reds", vmin=0, vmax=np.max(std))

    cb = plt.colorbar(sig_plot, ax = axs[2], orientation="vertical")
    cb.set_label('Uncertainty')
    plt.tight_layout()
    plt.grid()
    axs[2].grid(True)
    axs[2].set_ylabel("Repetitions")


    #real om
    dmu = oms 
    dmu_plot = axs[3].pcolormesh(dmu, cmap="bwr", vmin=-np.max(np.abs(dmu)), vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[3], orientation="vertical")
    cb.set_label('Real $\omega$')
    axs[3].grid(True)
    plt.tight_layout()
    plt.grid()
    axs[3].set_ylabel("Repetitions")

    #error
    dom = oms - mus
    om_plot = axs[4].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[4], orientation="vertical")
    cb.set_label('Est. error')
    plt.tight_layout()
    axs[4].grid(True)
    plt.ylabel("Repetitions")

    #error
    dom = np.abs(oms) - np.abs(mus)
    om_plot = axs[5].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[5], orientation="vertical")
    cb.set_label('Est. error')
    plt.tight_layout()
    axs[5].grid(True)
    plt.savefig("figures/games"+str(string)+".png")
    plt.xlabel("Consequtive shots (time)")
    plt.ylabel("Repetitions")

def analyse_few_games4(results, string):
    rewards, actions, oms, mus, std = import_data(results)
    rewards_to_plot = np.zeros(rewards.shape)
    times = np.zeros(rewards.shape)
    rewards_to_plot[rewards<0] = -1  #0 - success, #-1 -failure, times...
    rewards_to_plot[rewards==1] = 1
    times[rewards==0] = actions[rewards==0]
    maxtime = np.max(times)
    rewards_to_plot[rewards_to_plot==0] = None
    times[times==0] = None

    #est_prob = np.sum(outcome==2,axis=1)/len(outcome[0,:])
    #check_prob = np.sum(outcome==4,axis=1)/len(outcome[0,:])
    #tot_reward = np.sum(rewards,axis=1)
    std = std*mus
    #  Plot
    fig, axs = plt.subplots(6, 1, figsize=(6, 6), sharex=True)
    plt.subplots_adjust(hspace=0.0)
    from matplotlib import colors
    #axs[0].pcolormesh(actions, cmap='binary')
    #axs[0].set_title('Actions')
    # use the cmap which has blue for negative and red for positive values

    #plot rewards with -1 -> blue, 0-> red, and >1 -> colormap
    for k in range(6):
        axs[k].set_yticks([0,1,2,3])


    cmap_rewards = "bwr"
    cmap_times = "binary"

    #colors = ["b","r","k","w"]
    d = axs[0].pcolormesh(rewards_to_plot, cmap=cmap_rewards)
    d2 = axs[0].pcolormesh(times, cmap=cmap_times,vmax = maxtime*2)
    axs[0].grid(True)
    cb = plt.colorbar(d2, ax = axs[0], orientation="vertical")
    cb.set_label('Est. time')
    cb.set_ticks([])
    plt.tight_layout()
    axs[0].grid(True)
    axs[0].set_ylabel("Repetitions")
    #mu
    dmu = mus
    
    dmu_plot = axs[1].pcolormesh(dmu, cmap="Reds", vmin=0, vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[1], orientation="vertical")
    cb.set_label('Est. $\omega$')
    plt.tight_layout()
    axs[1].grid(True)
    axs[1].set_ylabel("Repetitions")


    #Ustd
    sig_plot = axs[2].pcolormesh(std, cmap="Reds", vmin=0, vmax=np.max(std))

    cb = plt.colorbar(sig_plot, ax = axs[2], orientation="vertical")
    cb.set_label('Uncertainty')
    plt.tight_layout()
    plt.grid()
    axs[2].grid(True)
    axs[2].set_ylabel("Repetitions")


    #real om
    dmu = oms 
    dmu_plot = axs[3].pcolormesh(dmu, cmap="bwr", vmin=-np.max(np.abs(dmu)), vmax=np.max(np.abs(dmu)))

    cb = plt.colorbar(dmu_plot, ax = axs[3], orientation="vertical")
    cb.set_label('Real $\omega$')
    axs[3].grid(True)
    plt.tight_layout()
    plt.grid()
    axs[3].set_ylabel("Repetitions")

    #error
    dom = oms - mus
    om_plot = axs[4].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[4], orientation="vertical")
    cb.set_label('Est. error')
    plt.tight_layout()
    axs[4].grid(True)
    plt.ylabel("Repetitions")

    #error
    dom = np.abs(oms) - np.abs(mus)
    om_plot = axs[5].pcolormesh(dom, cmap="bwr", vmin=-np.max(np.abs(dom)), vmax=np.max(np.abs(dom)))

    cb = plt.colorbar(om_plot, ax = axs[5], orientation="vertical")
    cb.set_label('Est. error')
    plt.tight_layout()
    axs[5].grid(True)
   # plt.savefig("figures/games"+str(string)+".png")
    plt.xlabel("Consequtive shots (time)")
    plt.ylabel("Repetitions")

    plt.savefig(string)


def analyse_decisions(results, string):
    rewards, actions, oms, mus, std = import_data(results)    
    outcome = np.zeros(mus.shape)
    outcome[rewards==1] = 1 
    outcome[rewards<0] = 0
    outcome[rewards==0] = 2
    
    import matplotlib.colors as colors
    cmap = colors.ListedColormap(['b','r','w',"k"])
    bounds = [0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    c = outcome.flatten()
    y = std.flatten()
    x = mus.flatten()
    z = y
    stringout = ""
    stringout += "Probability of estimation, success, failing, success given flip, reward \n"
    stringout += str(np.sum(outcome==2)/len(outcome.flatten())) + ","
    stringout += str(np.sum(outcome==1)/len(outcome.flatten())) + ","
    stringout += str(np.sum(outcome==0)/len(outcome.flatten())) + ","
    stringout += str(np.sum(outcome==1)/np.sum(outcome<2))+ ","
    stringout += str(np.average(np.sum(rewards,axis=1)))+ r'$ \pm $' + str(np.std(np.sum(rewards,axis=1)))


    #save string as txt
    with open(string+"stats.csv", "w") as text_file:
        text_file.write(stringout)





    plt.title("Actions")
    plt.scatter(x,z,c=c, marker=".", cmap = cmap, norm=norm, alpha=1, s=1)
    plt.legend()
    plt.xlabel("Estimated om")
    plt.ylabel("Estimated std")
    plt.grid()
    plt.yscale("log")

    plt.savefig(string+"A.png")
    
    plt.figure()
    plt.hist(x[np.where(c==0)],color="b", bins = 100, alpha=0.6, density=True)
    plt.hist(x[np.where(c==1)],color="r", bins = 100, alpha=0.6, density=True)
    plt.hist(x[np.where(c==2)],color="k", bins = 100,alpha=0.6, density=True)
    plt.hist(x[np.where(c==4)],color="g", bins = 100,alpha=0.6, density=True)
    plt.xlabel("Estimated om $\mu$")
    plt.savefig(string+"B.png")
    plt.figure()
    plt.hist(z[np.where(c==0)],color="b",bins = 100,alpha=0.6, density=True)
    plt.hist(z[np.where(c==1)],color="r",bins = 100,alpha=0.6, density=True)
    plt.hist(z[np.where(c==2)],color="k", bins = 100,alpha=0.6, density=True)
    plt.hist(z[np.where(c==4)],color="g", bins = 100,alpha=0.6, density=True)
    plt.xlabel("Estimated $\sigma$")
    plt.savefig(string+"C.png")


def analyse_time(results, string):
    plt.figure()
    rewards, actions, oms, mus, std = import_data(results)
   
    #cmap = colors.ListedColormap(['b','r','w',"k"])
    bounds = [0,1,2,3]
    #norm = colors.BoundaryNorm(bounds, cmap.N)
    c = actions.flatten()
    y = std.flatten()
    x = mus.flatten()
    z = y

    fig, ax = plt.subplots(4,1,figsize=(8,8))
    
    ax[0].hist(actions.flatten(), bins=11)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Number of estimations")



    a = ax[1].scatter(x,z,c=c, marker=".", alpha=0.5)
    ax[1].legend()
    ax[1].set_xlabel("Estimated om")
    ax[1].set_ylabel("Estimated std")
    ax[1].grid()
    ax[1].set_yscale("log")
    cb = plt.colorbar(a, ax=ax[1])

    #plt.savefig("figures/decisions_"+str(string)+".png")
    
    bins = ax[3].hist(x,bins=6)
    times = []
    for n,bin0 in enumerate(bins[1]):
        if n == len(bins[1])-1:
            break
        else:
            filtr = (x>bin0) * (x < bins[1][n+1])
            times.append(c[filtr])
    ax[2].violinplot(times, showmeans=True)
    
    bins = ax[3].hist(y,bins=11)
    times = []
    for n,bin0 in enumerate(bins[1]):
        if n == len(bins[1])-1:
            break
        else:
            filtr = (y>bin0) * (y < bins[1][n+1])
            times.append(c[filtr])
    ax[3].clear()
    plt.savefig(string)
    #ax[3].violinplot(times, showmeans=True)
    

def plot_learning_curve(path):
    #load monitor csv using numpy

    npzfile = np.load(path+"evaluations.npz")
    # extract the two arrays
    x = npzfile['results']

    avgs = np.average(x,axis=1)
    stds = np.std(x, axis=1)
    #plot the data
    plt.errorbar(range(len(avgs)), avgs, yerr=stds)
    plt.plot(range(len(avgs)),moving_average(avgs, 4))
    plt.xlabel("Number of steps")
    plt.ylabel("Reward")
    plt.savefig(path+"learning_curve.png")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

    