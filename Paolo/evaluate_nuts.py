import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Output folder
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_nuts_stats3") 

# CSV Filename
CSV_FILENAME = "results_nuts_batch.csv"
CSV_PATH = os.path.join(RESULTS_DIR, CSV_FILENAME)

# Number of classes
NUM_CLASSES = 7

# ==========================================
# 2. PARSING UTILS
# ==========================================
def parse_vector_col(df, col_name):
    """Converts CSV strings back into numpy arrays."""
    try:
        return df[col_name].apply(lambda x: np.fromstring(
            str(x).replace('[','').replace(']','').replace('\n',' ').replace(',', ' '), 
            sep=' '
        ))
    except Exception as e:
        print(f"Error parsing column {col_name}: {e}")
        return None

def get_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred), # Nota: questo è R2 vettoriale
        "CosSim": np.mean(cosine_similarity(y_true, y_pred).diagonal())
    }

def calculate_smape(y_true, y_pred):
    """Calcola SMAPE gestendo la divisione per zero."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2 + 1e-10
    return np.mean(numerator / denominator) * 100.0

def analyze_realnvp_impact(y_true, y_cvae_init, y_realnvp_refined):
    """
    y_true: One-hot vectors o indici reali
    y_cvae_init: Output grezzo del CVAE (vettori)
    y_realnvp_refined: Output finale del RealNVP (Mean o Mode vector)
    """
    # 1. Convertiamo tutto in indici di classe (0-6)
    true_cls = np.argmax(y_true, axis=1)
    cvae_cls = np.argmax(y_cvae_init, axis=1)
    nvp_cls  = np.argmax(y_realnvp_refined, axis=1)
    
    # 2. Calcolo Accuratezze
    acc_cvae = accuracy_score(true_cls, cvae_cls)
    acc_nvp  = accuracy_score(true_cls, nvp_cls)
    
    # 3. Matrice di Migrazione
    # Caso A: CVAE Sbaglia -> NVP Giusto (Salvato)
    saved_mask = (cvae_cls != true_cls) & (nvp_cls == true_cls)
    n_saved = np.sum(saved_mask)
    
    # Caso B: CVAE Giusto -> NVP Sbaglia (Peggiorato)
    degraded_mask = (cvae_cls == true_cls) & (nvp_cls != true_cls)
    n_degraded = np.sum(degraded_mask)
    
    # Caso C/D: Entrambi uguali
    both_correct = np.sum((cvae_cls == true_cls) & (nvp_cls == true_cls))
    both_wrong   = np.sum((cvae_cls != true_cls) & (nvp_cls != true_cls))
    
    # 4. Analisi Quantificazione (Solo sui casi dove la classe è giusta per entrambi)
    # Vogliamo vedere se affina il danno
    valid_mask = (cvae_cls == true_cls) & (nvp_cls == true_cls)
    if np.sum(valid_mask) > 0:
        # Estraiamo i livelli di danno scalari per la classe corretta
        # y_true[i, true_cls[i]] prende il valore 0.XX della cella giusta
        lev_true = y_true[valid_mask, true_cls[valid_mask]]
        lev_cvae = y_cvae_init[valid_mask, true_cls[valid_mask]]
        lev_nvp  = y_realnvp_refined[valid_mask, true_cls[valid_mask]]
        
        mae_cvae = mean_absolute_error(lev_true, lev_cvae)
        mae_nvp  = mean_absolute_error(lev_true, lev_nvp)
        quant_improvement = (mae_cvae - mae_nvp) / mae_cvae * 100
    else:
        mae_cvae, mae_nvp, quant_improvement = 0, 0, 0

    print("="*60)
    print(f"       ANALISI IMPATTO REALNVP (CVAE vs NVP)")
    print("="*60)
    print(f"Accuratezza CVAE : {acc_cvae:.2%}")
    print(f"Accuratezza NVP  : {acc_nvp:.2%}  (Delta: {acc_nvp-acc_cvae:+.2%})")
    print("-" * 60)
    print(f"✅ RECUPERATI (CVAE Err -> NVP Ok) : {n_saved} campioni")
    print(f"❌ PERSI      (CVAE Ok  -> NVP Err) : {n_degraded} campioni")
    print(f"⚖️  BILANCIO NETTO                 : {n_saved - n_degraded:+d} campioni")
    print("-" * 60)
    print(f"Analisi Quantificazione (sui casi corretti):")
    print(f"  - MAE Errore CVAE: {mae_cvae:.5f}")
    print(f"  - MAE Errore NVP : {mae_nvp:.5f}")
    print(f"  - Miglioramento  : {quant_improvement:+.2f}%")
    print("="*60)
    
    return saved_mask, degraded_mask

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
    'font.size': 16,          # Dimensione base
    'axes.titlesize': 20,     # Titolo del grafico
    'axes.labelsize': 20,     # Etichette assi (x, y, z)
    'xtick.labelsize': 14,    # Numeri asse x
    'ytick.labelsize': 14,    # Numeri asse y
    'legend.fontsize': 18,    # Testo legenda
    'figure.titlesize': 20    # Titolo figura globale
})
    
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] File not found: {CSV_PATH}")
        print("Please run 'run_nuts.py' first and ensure the 'R_hat' column is saved.\n")
        exit()

    print(f"--- Reading data from: {CSV_FILENAME} ---")
    df = pd.read_csv(CSV_PATH)
    
    # Check for R-hat column
    if 'R_hat' not in df.columns:
        print("WARNING: 'R_hat' column missing in CSV. Plot 9 will be empty.")
        df['R_hat'] = 0.0

    # ------------------------------------------
    # A. VECTOR PARSING
    # ------------------------------------------
    y_true = np.stack(parse_vector_col(df, 'True_Vector').values)
    y_mean = np.stack(parse_vector_col(df, 'Mean_Vector').values)
    y_mode = np.stack(parse_vector_col(df, 'Mode_Vector').values)
    y_std  = np.stack(parse_vector_col(df, 'Std_Vector').values) 
    y_vae  = np.stack(parse_vector_col(df, 'VAE_Init').values)
    r_hat_vals = df['R_hat'].values




    # ------------------------------------------
    # B. DATA PREPARATION 
    # ------------------------------------------
    print("Applying Hard Thresholding (< 0.05) to predictions...")

    def apply_threshold_keep_winner(y_matrix, threshold=0.05):
        """
        Azzera i valori sotto 'threshold', MA mantiene sempre
        il valore massimo (la classe vincente) anche se è sotto soglia.
        """
        y_clean = y_matrix.copy()
        # Trova l'indice del massimo per ogni riga (la classe scelta)
        max_indices = np.argmax(y_clean, axis=1)
        
        # Crea una maschera di chi è sotto soglia
        low_mask = y_clean < threshold
        
        # Applica la soglia (azzera)
        y_clean[low_mask] = 0.0
        
        # Ripristina il vincitore originale (per non perdere la classificazione)
        # Se il vincitore era < 0.05, viene comunque preservato
        rows = np.arange(len(y_clean))
        y_clean[rows, max_indices] = y_matrix[rows, max_indices]
        
        return y_clean

    # Applica il filtro ai vettori predetti
    y_mean = apply_threshold_keep_winner(y_mean, threshold=0.05)
    y_mode = apply_threshold_keep_winner(y_mode, threshold=0.05)
    
    idx_range = np.arange(len(y_true))
    true_cls_idx = np.argmax(y_true, axis=1)
    mean_cls_idx = np.argmax(y_mean, axis=1)
    mode_cls_idx = np.argmax(y_mode, axis=1)

    # Label 1-7
    true_cls = true_cls_idx + 1
    mean_cls = mean_cls_idx + 1
    mode_cls = mode_cls_idx + 1

    
    # Damage Levels (Scalar)
    true_levels = y_true[idx_range, true_cls_idx]
    vae_levels = y_vae[idx_range, true_cls_idx]

    # Predicted Levels (Scalar based on Predicted Class)
    pred_levels_mean_selected = y_mean[idx_range, mean_cls_idx]
    pred_levels_mode_selected = y_mode[idx_range, mode_cls_idx]

    # CVAE Predicted Class and Levels
    vae_cls_idx = np.argmax(y_vae, axis=1)
    pred_levels_cvae_selected = y_vae[idx_range, vae_cls_idx]

    # NUOVO: MAE scalare per il CVAE
    mae_scalar_level_cvae = mean_absolute_error(true_levels, pred_levels_cvae_selected)
    

    # Sample-wise metrics (Vector MAE)
    mae_sample_mean = np.mean(np.abs(y_true - y_mean), axis=1)
    mae_sample_mode = np.mean(np.abs(y_true - y_mode), axis=1)
    
    # Uncertainty
    avg_uncertainty = np.mean(y_std, axis=1)
    pred_uncertainty_mode = y_std[idx_range, mode_cls_idx]

    # Correct/Wrong mask (Using MEAN for primary plots)
    mask_ok_mean = (true_cls == mean_cls)
    mask_ok_mode = (true_cls == mode_cls) # For Mode Analysis
    
    exec_times = df['Time'].values if 'Time' in df.columns else None

    # ------------------------------------------
    # C. CALCOLO METRICHE ESTESE E TOP 20% INCERTI
    # ------------------------------------------
    met_mean = get_metrics(y_true, y_mean)
    met_mode = get_metrics(y_true, y_mode)
    acc_mean = accuracy_score(true_cls, mean_cls)
    acc_mode = accuracy_score(true_cls, mode_cls)

    # MSE e RMSE (Vettoriali)
    mse_vec_mean = mean_squared_error(y_true, y_mean)
    mse_vec_mode = mean_squared_error(y_true, y_mode)
    
    # SMAPE (Vettoriale)
    smape_vec_mean = calculate_smape(y_true, y_mean)
    smape_vec_mode = calculate_smape(y_true, y_mode)

    # R2 Regression (Scalar vs Scalar - Damage Level)
    r2_reg_mean = r2_score(true_levels, pred_levels_mean_selected)
    r2_reg_mode = r2_score(true_levels, pred_levels_mode_selected)

    # Calcolo MAE scalare (solo sul livello di danno nella classe predetta vs classe reale)
    mae_scalar_level_mean = mean_absolute_error(true_levels, pred_levels_mean_selected)
    mae_scalar_level_mode = mean_absolute_error(true_levels, pred_levels_mode_selected)

    
    
    # Cosine Similarity (Vector)
    cos_sim_mean = met_mean['CosSim']
    cos_sim_mode = met_mode['CosSim']

    # Average Exec Time
    avg_exec_time = np.mean(exec_times) if exec_times is not None else 0.0

    # --- TOP 20% UNCERTAIN SAMPLES CALCULATION ---
    unc_threshold = np.percentile(avg_uncertainty, 80)
    top20_mask = avg_uncertainty >= unc_threshold
    num_top20 = np.sum(top20_mask)
    
    acc_top20_mean = accuracy_score(true_cls[top20_mask], mean_cls[top20_mask])
    acc_top20_mode = accuracy_score(true_cls[top20_mask], mode_cls[top20_mask])
    
    mae_top20_vec_mean = np.mean(mae_sample_mean[top20_mask])
    mae_top20_vec_mode = np.mean(mae_sample_mode[top20_mask])
    
    mae_top20_int_mean = mean_absolute_error(true_levels[top20_mask], pred_levels_mean_selected[top20_mask])
    mae_top20_int_mode = mean_absolute_error(true_levels[top20_mask], pred_levels_mode_selected[top20_mask])


    print("\n" + "="*60)
    print(f"NUTS EVALUATION REPORT")
    print("-" * 60)
    print(f"{'METRICA':<15} | {'MEAN Est':<12} | {'MODE Est':<12}")
    print("-" * 60)
    print(f"Accuracy        | {acc_mean:.2%}      | {acc_mode:.2%}")
    print(f"MAE (Vector)    | {met_mean['MAE']:.5f}      | {met_mode['MAE']:.5f}")
    print(f"R2 (Regression) | {r2_reg_mean:.5f}      | {r2_reg_mode:.5f}")
    print(f"Avg Time        | {avg_exec_time:.2f} s")
    print("="*60)

    # ==========================================
    # D. GRAPHICS (10 PLOTS)
    # ==========================================

    # --- PLOT 1: Confusion Matrix (Mean) ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_cls, mean_cls, labels=range(1, NUM_CLASSES + 1))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=range(1, NUM_CLASSES + 1), yticklabels=range(1, NUM_CLASSES + 1))
    plt.title(f'Confusion Matrix - NUTS (MEAN Strategy)\nAcc: {acc_mean:.2%}')
    plt.xlabel('Predicted Class'); plt.ylabel('True Class')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_01_cm_mean.png")); plt.close()

    # --- PLOT 2: Regression (Mode - Reference) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok_mode], pred_levels_mode_selected[mask_ok_mode], alpha=0.6, c='dodgerblue', edgecolor='k', label='Correct Classification')
    plt.scatter(true_levels[~mask_ok_mode], pred_levels_mode_selected[~mask_ok_mode], alpha=0.8, c='crimson', marker='X', label='Wrong Classification')
    plt.plot([0, 0.6], [0, 0.6], 'k--', label='Ideal Fit')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Level (Mode)')
    plt.legend()
    plt.title(f'Regression Analysis (Mode)\nR2: {r2_reg_mode:.3f} | MAE: {np.mean(met_mode["MAE"]):.4f}')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_02_reg_mode.png")); plt.close()

    # --- PLOT 3: Regression (Mean - Main) ---
    plt.figure(figsize=(9, 8))
    plt.scatter(true_levels[mask_ok_mean], pred_levels_mean_selected[mask_ok_mean], alpha=0.6, c='mediumseagreen', edgecolor='k', label='Correct Classification')
    plt.scatter(true_levels[~mask_ok_mean], pred_levels_mean_selected[~mask_ok_mean], alpha=0.8, c='crimson', marker='X', label='Wrong Classification')
    plt.plot([0, 0.6], [0, 0.6], 'k--', label='Ideal Fit'); plt.legend()
    plt.title(f'Regression Damage Analysis')
    plt.xlabel('True Damage Level'); plt.ylabel('Predicted Damage Level')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_03_reg_mean.png")); plt.close()

    # --- PLOT 4: Head-to-Head Error ---
    plt.figure(figsize=(9, 8))
    max_err = max(mae_sample_mean.max(), mae_sample_mode.max()) * 1.05
    plt.scatter(mae_sample_mean, mae_sample_mode, alpha=0.6, c='purple', edgecolor='k')
    plt.plot([0, max_err], [0, max_err], 'r--', label='Identity Line')
    plt.xlabel('MAE (Mean Strategy)'); plt.ylabel('MAE (Mode Strategy)')
    plt.title('Error Comparison: Mean vs Mode')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_04_error_comp.png")); plt.close()

    # --- PLOT 5: VAE Shift (Mean) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(true_levels, vae_levels, marker='x', c='gray', alpha=0.5, label='Initial VAE Pred')
    plt.scatter(true_levels, pred_levels_mean_selected, marker='o', c='orange', edgecolor='k', label='Final NUTS Mean')
    for i in range(len(true_levels)):
        if abs(pred_levels_mean_selected[i] - vae_levels[i]) > 0.05:
            plt.plot([true_levels[i], true_levels[i]], [vae_levels[i], pred_levels_mean_selected[i]], c='gray', alpha=0.2)
    plt.plot([0, 0.6], [0, 0.6], 'k--', label='Ideal Fit'); plt.legend()
    plt.xlabel('True Level'); plt.ylabel('Predicted Level')
    plt.title('Refinement Analysis: VAE Initial vs NUTS Final Mean')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_05_vae_shift.png")); plt.close()

    # --- PLOT 6: Calibration (Uncertainty vs Error) ---
    plt.figure(figsize=(10, 8))
    plt.scatter(avg_uncertainty[mask_ok_mean], mae_sample_mean[mask_ok_mean], c='mediumseagreen', label='Correct')
    plt.scatter(avg_uncertainty[~mask_ok_mean], mae_sample_mean[~mask_ok_mean], c='crimson', marker='X', label='Wrong')
    if len(avg_uncertainty) > 1:
        z = np.polyfit(avg_uncertainty, mae_sample_mean, 1); p = np.poly1d(z)
        plt.plot(avg_uncertainty, p(avg_uncertainty), "k--", alpha=0.6, label='Trendline')
    plt.xlabel('Uncertainty (Avg Std Dev)'); plt.ylabel('Error (Sample MAE Mean)'); plt.legend()
    plt.title('Calibration Check: Uncertainty vs Error')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_06_calibration.png")); plt.close()

    # --- PLOT 7: Uncertainty Boxplot ---
    plt.figure(figsize=(8, 6))
    data_unc = [avg_uncertainty[mask_ok_mean], avg_uncertainty[~mask_ok_mean]]
    if len(avg_uncertainty[~mask_ok_mean]) > 0:
        bplot = plt.boxplot(data_unc, patch_artist=True, tick_labels=['Correct', 'Wrong'])
        for patch, color in zip(bplot['boxes'], ['mediumseagreen', 'crimson']): patch.set_facecolor(color)
    plt.title('Uncertainty Distribution vs Correctness')
    plt.ylabel('Average Standard Deviation')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_07_unc_boxplot.png")); plt.close()

    # --- PLOT 8: Uncertainty per Class ---
    plt.figure(figsize=(10, 6))
    avg_unc_per_class = []
    class_labels = range(1, NUM_CLASSES + 1)
    counts = []
    
    for c in class_labels:
        mask_c = (mean_cls == c)
        if np.sum(mask_c) > 0:
            avg_unc_per_class.append(np.mean(avg_uncertainty[mask_c]))
            counts.append(np.sum(mask_c))
        else:
            avg_unc_per_class.append(0)
            counts.append(0)

    bars = plt.bar(class_labels, avg_unc_per_class, color=sns.color_palette("viridis", NUM_CLASSES), edgecolor='k')
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        if h > 0: plt.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)

    plt.title('Average Uncertainty per Predicted Class (Mean Strategy)')
    plt.xlabel('Predicted Class'); plt.ylabel('Avg Std Dev')
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_08_unc_class.png")); plt.close()

    # --- PLOT 9: R-hat Boxplot (Correct vs Wrong) ---
    plt.figure(figsize=(8, 6))
    
    rhat_correct = r_hat_vals[mask_ok_mean]
    rhat_wrong   = r_hat_vals[~mask_ok_mean]
    
    data_rhat = [rhat_correct, rhat_wrong]
    labels_rhat = [f'Correct (n={len(rhat_correct)})', f'Wrong (n={len(rhat_wrong)})']
    
    if len(rhat_wrong) > 0:
        bplot = plt.boxplot(data_rhat, patch_artist=True, tick_labels=labels_rhat)
        colors = ['mediumseagreen', 'crimson']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    else:
        plt.boxplot([rhat_correct], patch_artist=True, tick_labels=['Correct Only'])
        plt.text(0.5, 0.5, "Perfect Accuracy!", ha='center')

    plt.axhline(1.01, color='red', linestyle='--', linewidth=1.5, label='Threshold 1.01 (Ideal)')
    plt.title('Convergence Quality (R-hat) vs Classification Accuracy', fontsize=14)
    plt.ylabel('R-hat Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, axis='y', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_09_rhat_boxplot.png"))
    plt.close()

    # --- PLOT 10: MAE per Class and Global MAE ---
    plt.figure(figsize=(10, 6))
    
    mae_per_class = []
    unc_per_class_mean = [] # Serve per il report testuale
    
    for c in class_labels:
        mask_true_c = (true_cls == c)
        if np.sum(mask_true_c) > 0:
            mae_c = np.mean(np.abs(y_true[mask_true_c] - y_mean[mask_true_c]))
            unc_c = np.mean(avg_uncertainty[mask_true_c])
            mae_per_class.append(mae_c)
            unc_per_class_mean.append(unc_c)
        else:
            mae_per_class.append(0)
            unc_per_class_mean.append(0)

    mae_global = met_mean['MAE'] 
    bars = plt.bar(class_labels, mae_per_class, color=sns.color_palette("muted", NUM_CLASSES), edgecolor='k', alpha=0.8, label='MAE per Class')
    plt.axhline(mae_global, color='black', linestyle='--', linewidth=2, label=f'Global Avg MAE: {mae_global:.4f}')

    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title('Mean Absolute Error (MAE) per True Class', fontsize=14)
    plt.xlabel('True Class', fontsize=12); plt.ylabel('MAE', fontsize=12)
    plt.xticks(class_labels); plt.legend(); plt.grid(True, axis='y', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "nuts_beam_10_mae_per_class.png"))
    plt.close()

    print(f"\n[DONE] Generated 10 plots in:\n{RESULTS_DIR}")

    # ==========================================
    # E. SCRITTURA FILE TESTUALE
    # ==========================================
    SUMMARY_PATH = os.path.join(RESULTS_DIR, "summary_statistics.txt")
    
    # Dati Refinement
    vae_cls_idx = np.argmax(y_vae, axis=1)
    vae_cls = vae_cls_idx + 1
    improved_mask = (vae_cls != true_cls) & (mean_cls == true_cls)
    n_improved = np.sum(improved_mask)
    degraded_mask = (vae_cls == true_cls) & (mean_cls != true_cls)
    n_degraded = np.sum(degraded_mask)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("===================================================================\n")
        f.write("             RIEPILOGO STATISTICO COMPLETO NUTS\n")
        f.write("===================================================================\n\n")
        
        f.write(f"Timestamp: {pd.Timestamp.now()}\n")
        f.write(f"Mae: {np.mean(met_mean['MAE'])}\n")
        f.write(f"R2: {np.mean(met_mean['R2'])}\n")
        f.write(f"Campioni: {len(df)}\n")
        f.write(f"Tempo Medio: {avg_exec_time:.4f} s\n")
        f.write(f"Incertezza Globale: {np.mean(y_std):.6f}\n")
        f.write(f"R-hat Medio: {np.mean(r_hat_vals):.4f} (Max: {np.max(r_hat_vals):.4f})\n\n")
        
        f.write("-" * 85 + "\n")
        f.write(f"{'METRICA':<15} | {'MEAN Est':<25} | {'MODE Est':<25}\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'MAE (Vector)':<15} | {met_mean['MAE']:.6f}                  | {met_mode['MAE']:.6f}\n")
        f.write(f"{'MAE (Intensity)':<15} | {mae_scalar_level_mean:.6f}                  | {mae_scalar_level_mode:.6f}\n")
        f.write(f"{'CVAE MAE (Intensity)':<15} | {mae_scalar_level_cvae:.6f}                  | -- \n")
        f.write(f"{'MSE (Vector)':<15} | {mse_vec_mean:.6f}                  | {mse_vec_mode:.6f}\n")
        f.write(f"{'RMSE (Vector)':<15} | {np.sqrt(mse_vec_mean):.6f}                  | {np.sqrt(mse_vec_mode):.6f}\n")
        f.write(f"{'R2 (Regression)':<15} | {r2_reg_mean:.6f}                  | {r2_reg_mode:.6f}\n")
        f.write(f"{'Cos Sim':<15} | {cos_sim_mean:.6f}                  | {cos_sim_mode:.6f}\n")
        f.write(f"{'SMAPE':<15} | {smape_vec_mean:.6f}                  | {smape_vec_mode:.6f}\n")
        f.write("-" * 85 + "\n\n")
        
        f.write("--- CLASSIFICAZIONE ---\n")
        f.write(f"Acc (Mean): {acc_mean:.2%} ({int(acc_mean*len(df))}/{len(df)})\n")
        f.write(f"Acc (Mode): {acc_mode:.2%} ({int(acc_mode*len(df))}/{len(df)})\n\n")
        
        f.write("--- DETTAGLIO PER CLASSE REALE (Mean Est) ---\n")
        f.write(f"{'Class':<6} | {'Avg MAE':<12} | {'Avg Unc (Std)':<22} | {'Count':<6}\n")
        f.write("-" * 60 + "\n")
        for i, c in enumerate(class_labels):
            count_c = np.sum(true_cls == c)
            f.write(f"{c:<6} | {mae_per_class[i]:.6f}      | {unc_per_class_mean[i]:.6f}                 | {count_c:<6}\n")
        f.write("\n")
        
        # --- NEW SECTION: TOP 20% UNCERTAIN SAMPLES ---
        f.write("--- ANALISI DEL 20% DEI CAMPIONI PIÙ INCERTI (Top 20% Avg Std) ---\n")
        f.write(f"Soglia Incertezza (80° percentile): {unc_threshold:.6f}\n")
        f.write(f"Numero di campioni valutati: {num_top20}\n\n")
        
        f.write(f"{'METRICA':<20} | {'MEAN Estimator':<20} | {'MODE Estimator':<20}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Accuracy':<20} | {acc_top20_mean:<20.2%} | {acc_top20_mode:<20.2%}\n")
        f.write(f"{'MAE (Vector)':<20} | {mae_top20_vec_mean:<20.6f} | {mae_top20_vec_mode:<20.6f}\n")
        f.write(f"{'MAE (Intensity)':<20} | {mae_top20_int_mean:<20.6f} | {mae_top20_int_mode:<20.6f}\n")
        f.write("-" * 65 + "\n\n")

        f.write("--- REFINEMENT (CVAE vs NUTS Mean) ---\n")
        f.write(f"Migliorati (CVAE err -> NUTS ok): {n_improved}\n")
        f.write(f"Peggiorati (CVAE ok  -> NUTS err): {n_degraded}\n")
        f.write(f"Netto: {n_improved - n_degraded:+d}\n\n")

        f.write("--- ERRORI CLASSIFICAZIONE (MODE) ---\n")
        mask_wrong = (true_cls != mode_cls)
        if np.sum(mask_wrong) > 0:
            f.write(f"Totale Errori: {np.sum(mask_wrong)}\n")
            f.write(f"{'Index':<8} | {'True':<6} | {'Pred':<6} | {'MAE (Mode)':<12} | {'Uncertainty':<12} | {'R-hat':<10}\n")
            f.write("-" * 75 + "\n")
            
            idxs = df.index[mask_wrong]
            t_wrong = true_cls[mask_wrong]
            p_wrong = mode_cls[mask_wrong]
            mae_wrong = mae_sample_mode[mask_wrong]
            unc_wrong_spec = pred_uncertainty_mode[mask_wrong]
            rh_wrong = r_hat_vals[mask_wrong]
            
            for idx_v, t, p, m, u, r in zip(idxs, t_wrong, p_wrong, mae_wrong, unc_wrong_spec, rh_wrong):
                 f.write(f"{df.iloc[idx_v]['Index']:<8} | {t:<6} | {p:<6} | {m:.6f}       | {u:.6f}       | {r:.4f}\n")
            f.write(f"{'MAE (Intensity)':<15} | {mae_scalar_level_mean:.6f}                  | {mae_scalar_level_mode:.6f}\n")

        else:
            f.write("Nessun errore.\n")

    print(f"Statistiche complete salvate in: {SUMMARY_PATH}")
    
    # Usiamo y_mean come output raffinato principale del NUTS
    print("\n--- ESECUZIONE ANALISI DI REFINEMENT ---")
    analyze_realnvp_impact(y_true, y_vae, y_mean)