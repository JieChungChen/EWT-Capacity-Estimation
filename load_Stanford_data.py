import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from ewtpy import EWT1D
CELL_IDS = ['G1', 'V4', 'V5', 'W4', 'W5', 'W8', 'W9', 'W10']


def get_mat_voltage(filepath, plot=False):
    """
    get the voltage info. from .mat files
    sampling rate: 10Hz
    """
    f = loadmat(filepath)
    # print(f.keys())
    current = f['I_full_vec_M1_NMC25degC']
    time = f['t_full_vec_M1_NMC25degC']
    voltage = f['V_full_vec_M1_NMC25degC']
    step = f['Step_Index_full_vec_M1_NMC25degC']
    if plot:
        plt.plot(step, c='black', label='step')
        plt.plot(voltage, c='blue', label='voltage')
        plt.xlabel('time(s)')
        plt.legend()
        plt.show()
        plt.close()
    return voltage, step


def mat_to_npy(folder='Stanford_Dataset', sep=10):
    """
    W3、W7 cell量測上有問題, 因此排除在使用資料外
    folder: string, 放置cycling資料的位置
    sep: int, 採樣間隔
    """
    # 去除有問題的cycle
    ignore_dict = {'G1':[29], 'V4':[69], 'V5':[], 'W4':[], 'W5':[192], 'W8':[], 'W9':[137, 166], 'W10':[]}
    for cell_id in CELL_IDS:
        dc_v = []
        for cycle in range(14):
            filepath = '%s/Cycling_%d/%s.mat' % (folder, cycle+1, cell_id)
            if os.path.exists(filepath):
                v, s = get_mat_voltage(filepath)
                v[s!=14] = 0 # remove non-discharging part
                dif = np.abs(np.diff(v, axis=0)) # calculate discharge boundaries
                edge = np.argwhere(dif>1)+1 # find discharge boundaries
                dc_v+=[v[edge[i, 0]:edge[i+1, 0]] for i in range(0, len(edge)-1, 2)] # separate discharge area, len(edge)-1 for odd number boundaries
                print('%s partial discharge segments=%d'%(cell_id, len(edge)//2))
        print('%s total discharge segments=%d'%(cell_id, len(dc_v)))

        for i, cycle_info in enumerate(dc_v):
            # cycle_info shape: (time_step, 1)
            if (i+1) not in ignore_dict[cell_id]:
                points = np.arange(0, len(cycle_info), sep)
                os.makedirs('%s/discharge_info/%s'%(folder, cell_id), exist_ok=True)
                np.save('%s/discharge_info/%s/cycle_%s.npy'%(folder, cell_id, str(i+1).zfill(3)), cycle_info[points])

    
def feature_extraction(v):
    v_abs = np.abs(v)

    v_peak = np.max(v_abs) # peaSk value
    sd = np.std(v) # standard deviation
    m = np.mean(v) # mean
    kur = np.mean((v-m)**4)/sd**4 # kurtosis  
    skew = np.mean((v-m)**3)/sd**3 # skewness
    rms = np.sqrt(np.mean(v**2)) # RMS
    snr =  m/sd # SNR
    cl_f = v_peak/((np.mean(np.sqrt(v_abs)))**2) # clearance factor
    cr_f = v_peak/np.sqrt(np.mean(v**2)) # crest factor
    ip_f = v_peak/np.mean(v_abs) # impulse factor   
    sh_f = rms/np.mean(v_abs) # shape factor
    return [cl_f, cr_f, ip_f, kur, m, v_peak, rms, sh_f, skew, sd]


def npy_to_selected_features(folder):
    feature_names = ['Clearance Factor', 'Crest Factor', 'Impulse Factor', 'Kurtosis', 'Mean', 'Peak Value', 'RMS', 'Shape Factor', 'Skewness', 'SD']
    cell_id_list = os.listdir(folder)
    print(cell_id_list)
    for cell_id in cell_id_list:
        files = os.listdir(folder+'/'+cell_id)
        feature_list = []
        for f in files:
            v_t = np.load(folder+'/'+cell_id+'/'+f)
            v_t = v_t[0:36000, 0]
            ewt, mfb ,boundaries = EWT1D(v_t, N=9)
            corr = np.array([np.corrcoef(ewt[:, mode], v_t)[0, 1] for mode in range(9)])
            f_mode = ewt[:, np.argmax(corr)]
            v_c = v_t-f_mode 
            feature_list.append(feature_extraction(v_c))
        feature_list = np.vstack(feature_list)
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            ax[i//5,i%5].plot(feature_list[:, i])
            ax[i//5,i%5].set_title(feature_names[i], fontsize=8)
            ax[i//5,i%5].tick_params(axis='x', labelsize=5)
            ax[i//5,i%5].tick_params(axis='y', labelsize=5)
            ax[i//5,i%5].yaxis.get_offset_text().set(size=5)
        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        print(cell_id)
        plt.show()
        plt.close()


def v_curve_visualization(folder, cell_id):
    files = os.listdir(folder+'/'+cell_id)
    cmap = plt.get_cmap('coolwarm')
    for i, f in enumerate(files):
        v_curve = np.load(folder+'/'+cell_id+'/'+f)
        points = np.arange(0, len(v_curve), 10)
        plt.plot(points, v_curve[points], c=cmap(i/len(files)), alpha=0.7, lw=0.5)
        # plt.savefig('figs/%d.png'%(i+1))
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(files)),cmap='coolwarm')
    sm.set_array([])
    plt.colorbar(sm)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.show()
    plt.close()


def ewt_visualization():
    folder = 'Stanford_Dataset/discharge_info/W8'
    files = os.listdir(folder)
    v_t = np.load(folder+'/'+files[0])
    ewt, mfb ,boundaries = EWT1D(v_t[:, 0], N=9)

    fig, ax = plt.subplots(3, 3)

    for i in range(9):
        ax[i//3,i%3].plot(np.arange(len(v_t)), ewt[:, i])
        ax[i//3,i%3].set_title('Mode%d'%(i+1))
        ax[i//3,i%3].set_xlabel('Time (s)', fontsize=8)
        ax[i//3,i%3].set_ylabel('Voltage (V)', fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.close()
    
    # corr = np.array([np.corrcoef(ewt[:, mode], v_t[:, 0])[0, 1] for mode in range(9)])
    f_mode = ewt[:, 0]+ewt[:, 1]
    v_c = v_t[:,0]-f_mode
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(v_t, c='blue', ls='--', label='Voltage') # original signal
    ax[0].plot(f_mode, c='red', label='Mode 1') # fundamental mode
    ax[0].set_xlabel('Time(s)', fontsize=12)
    ax[0].set_ylabel('Voltage (V)', fontsize=12)
    ax[0].set_title('original voltage signal', fontsize=14)
    ax[0].legend()

    ax[1].plot(v_c, c='blue')
    ax[1].set_xlabel('Time(s)', fontsize=12)
    ax[1].set_ylabel('Voltage (V)', fontsize=12)
    ax[1].set_title('filtered signal', fontsize=14)
    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    # mat_to_npy('Stanford_Dataset')
    # v_curve_visualization('Stanford_Dataset/discharge_info', 'W9')
    # ewt_visualization()
    npy_to_selected_features('Stanford_Dataset/discharge_info')


if __name__ == '__main__':
    main()


