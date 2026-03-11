import glob
import os
import matplotlib.pyplot as plt
import pandas as pd

log_dir = '/pscratch/sd/s/sqian/part_training_output/scaling_study/slurm_logs/'
files = sorted(glob.glob(f'{log_dir}/slurm-4980*.out'))

def parse_logs(log_files):
    results = []
    
    # Rough parameter counts for sizes in Million
    params_map = {
        'nano': 0.1, 
        'tiny': 0.4,
        'small': 1.0,
        'base': 2.14
    }
    
    for f in log_files:
        job_id = f.split('-')[-1].split('.')[0]
        try:
            with open(f, 'r') as fh:
                lines = fh.readlines()
                model = "Unknown"
                budget = "Unknown"
                best = 0.0
                epoch = "0"
                for line in lines:
                    if "Model size:" in line: 
                        model = line.strip().split()[-1]
                    if "Data budget:" in line: 
                        budget = line.strip().split()[-1]
                    if 'Epoch #' in line and 'validating' in line:
                        epoch = line.split('Epoch #')[-1].split(' ')[0]
                    if "Current validation metric:" in line:
                        best_str = line.strip().split('best:')[-1].replace(')','').strip()
                        best_str = best_str.replace('\x1b[0m', '')
                        best = float(best_str)
                
                if model != "Unknown" and budget != "Unknown" and best > 0.0:
                    budget_m = float(budget.replace('M', '')) if 'M' in budget else float(budget)
                    results.append({
                        'job': job_id,
                        'model': model,
                        'budget_M': budget_m,
                        'params_M': params_map.get(model, 1.0),
                        'epoch': int(epoch),
                        'roc_auc': best
                    })
        except Exception as e:
            # Skip unreadable or incomplete files
            pass
            
    return results

def plot_scaling(results_data):
    df = pd.DataFrame(results_data)
    
    if len(df) == 0:
        print("No parsable summary data found. Jobs might still be initializing or failed.")
        return
        
    print("\nScaling Data Extracted:")
    print("-" * 50)
    print(df.sort_values(by=['budget_M', 'params_M']).to_string(index=False))
    print("-" * 50)
    
    import numpy as np
    from scipy.optimize import curve_fit
    
    # joint chinchilla form using independent variables N and D
    def chinchilla_form(X, E, A, B, alpha, beta):
        N, D = X
        # N is Model Parameters in M
        # D is Data Budget in M
        return E - A * (N ** (-alpha)) - B * (D ** (-beta))
        
    N_data = df['params_M'].values
    D_data = df['budget_M'].values
    AUC_data = df['roc_auc'].values
    
    popt = None
    try:
        popt, _ = curve_fit(
            chinchilla_form, 
            (N_data, D_data), 
            AUC_data, 
            p0=[0.9, 0.1, 0.1, 0.3, 0.3],
            bounds=([0.5, 0, 0, 0, 0], [1.0, 10, 10, 5, 5]),
            maxfev=50000
        )
        print(f"\nJoint Fit Parameters: E={popt[0]:.3f}, A={popt[1]:.3f}, B={popt[2]:.3f}, alpha={popt[3]:.3f}, beta={popt[4]:.3f}")
    except Exception as e:
        print(f"\nJoint Fit failed: {e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Performance vs Data size (fixed model constraints)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    models = ['nano', 'tiny', 'small', 'base']
    for i, model in enumerate(models):
        if model in df['model'].values:
            sub = df[df['model'] == model].sort_values('budget_M')
            label_str = f"{model} ({sub['params_M'].iloc[0]}M)"
            ax1.plot(sub['budget_M'], sub['roc_auc'], marker='o', linestyle='', color=colors[i], label=label_str)
            if popt is not None:
                x_fit = np.logspace(np.log10(sub['budget_M'].min()*0.8), np.log10(1000), 50)
                n_val = sub['params_M'].iloc[0]
                y_fit = chinchilla_form((n_val, x_fit), *popt)
                ax1.plot(x_fit, y_fit, color=colors[i], alpha=0.7, linestyle='--')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Data Budget (M samples)')
    ax1.set_ylabel('ROC-AUC')
    ax1.set_title('Performance vs Data Size\n$AUC = E - A \cdot N^{-\\alpha} - B \cdot D^{-\\beta}$')
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    if popt is not None:
        eq_str = f"Joint Fit:\n$E={popt[0]:.3f}$\n$A={popt[1]:.3f}$, $\\alpha={popt[3]:.3f}$\n$B={popt[2]:.3f}$, $\\beta={popt[4]:.3f}$"
        ax1.text(0.4, 0.05, eq_str, transform=ax1.transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 2. Performance vs Model size (fixed data budget constraints)
    budgets = sorted(df['budget_M'].unique())
    prop_cycle = plt.rcParams['axes.prop_cycle']
    opt_colors = prop_cycle.by_key()['color']
    for i, budget in enumerate(budgets):
        sub = df[df['budget_M'] == budget].sort_values('params_M')
        # Skip fitting if not enough points for variance
        c = opt_colors[i % len(opt_colors)]
        label_str = f"{budget}M Data"
        ax2.plot(sub['params_M'], sub['roc_auc'], marker='s', linestyle='', color=c, label=label_str)
        if popt is not None:
            x_fit = np.logspace(np.log10(0.05), np.log10(20.0), 50)
            d_val = budget
            y_fit = chinchilla_form((x_fit, d_val), *popt)
            ax2.plot(x_fit, y_fit, color=c, alpha=0.7, linestyle='--')
        
    ax2.set_xscale('log')
    ax2.set_xlabel('Model Parameters (M)')
    ax2.set_ylabel('ROC-AUC')
    ax2.set_title('Performance vs Model Size\n$AUC = E - A \cdot N^{-\\alpha} - B \cdot D^{-\\beta}$')
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    output_path = '/global/homes/s/sqian/jetclass_dir/particle_transformer/scaling/scaling_laws_fitted.png'
    plt.savefig(output_path, dpi=200)
    print(f"\nSaved scaling law plots to {output_path}")

if __name__ == '__main__':
    res = parse_logs(files)
    plot_scaling(res)
