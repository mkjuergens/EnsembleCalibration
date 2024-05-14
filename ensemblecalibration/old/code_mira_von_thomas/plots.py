df = pd.read_csv("./final_results_experiments_t1t2_alpha_100_10_10_0.01.csv")
methods = [r'SKCE$_{ul}$',r'HL(5)',r'HL(10)',r'ECE$_{conf}(5)$',r'ECE$_{conf}(10)$',r'ECE$_{cwise}$(5)',r'ECE$_{cwise}$(10)']
scenario = ["S1", "S2", "S3"]
alpha = np.array([0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.79, 0.87, 0.95])
fig, ax = plt.subplots(7,3,figsize=(7,10))
for i,m in enumerate(methods):
    for j,s in enumerate(scenario):
        #mean, std = get_mean_error(df, alpha, s, methods_idx[i])
        #ax[i,j].errorbar(alpha, mean, yerr=std)
        mean = ast.literal_eval(df.iloc[j,i])
        ax[i,j].plot(alpha, mean)
        ax[i,j].set_ylim([-0.05,1.05])
        ax[i,j].grid()
        if i==0:
            ax[i,j].set_title(s)
        if j==0:
            ax[i,j].plot(alpha,alpha,"r--",alpha=0.5)
            ax[i,j].set_ylabel(m)
        
fig.text(0.54, -0.01, 'significance level', va='center', ha='center', fontsize="large")
fig.text(-0.01, 0.5, 'empirical type I/II error rate', va='center', ha='center', rotation='vertical', fontsize="large")
plt.tight_layout()
#plt.savefig("./plot.pdf", dpi=300, bbox_inches = "tight")