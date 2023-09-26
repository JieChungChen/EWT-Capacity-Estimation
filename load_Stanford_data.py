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
    return voltage, current, step


def mat_to_npy(folder='Stanford_Dataset', sep=10):
    """
    W3、W7 cell量測上有問題, 因此排除在使用資料外
    folder: string, 放置cycling資料的位置
    sep: int, 採樣間隔
    """
    # 去除有問題的cycle
    ignore_dict = {'G1':[29], 'V4':[69], 'V5':[], 'W4':[], 'W5':[192], 'W8':[], 'W9':[137, 167], 'W10':[]}
    for cell_id in CELL_IDS:
        dc_v, dc_c = [], []
        for cycle in range(14):
            filepath = '%s/Cycling_%d/%s.mat' % (folder, cycle+1, cell_id)
            if os.path.exists(filepath):
                v, c, s = get_mat_voltage(filepath)
                v[s!=14] = 0 # remove non-discharging part
                dif = np.abs(np.diff(v, axis=0)) # calculate discharge boundaries
                edge = np.argwhere(dif>1)+1 # find discharge boundaries
                dc_v+=[v[edge[i, 0]:edge[i+1, 0]] for i in range(0, len(edge)-1, 2)] # separate discharge area, len(edge)-1 for odd number boundaries
                dc_c+=[c[edge[i, 0]:edge[i+1, 0]] for i in range(0, len(edge)-1, 2)]
                print('%s partial discharge segments=%d'%(cell_id, len(edge)//2))
        print('%s total discharge segments=%d'%(cell_id, len(dc_v)))

        for i in range(len(dc_v)):
            # cycle_info shape: (time_step, 2)
            v, c = dc_v[i], dc_c[i]
            cycle_info = np.hstack([v, c])
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


def npy_filtering(folder):
    cell_id_list = os.listdir(folder+'/discharge_info')
    for cell_id in cell_id_list:
        os.makedirs('%s/filtered_info/%s'%(folder, cell_id), exist_ok=True)
        files = os.listdir(folder+'/discharge_info/'+cell_id)
        for i, f in enumerate(files):
            v_t = np.load(folder+'/discharge_info/'+cell_id+'/'+f)
            ewt, mfb ,boundaries = EWT1D(v_t[:, 0], N=9)
            f_mode = ewt[:, 0]
            v_c = v_t[:,0]-f_mode
            np.save('%s/filtered_info/%s/cycle_%s.npy'%(folder, cell_id, str(i+1).zfill(3)), v_c)


def npy_to_selected_features(folder):
    feature_names = ['Clearance Factor', 'Crest Factor', 'Impulse Factor', 'Kurtosis', 'Mean', 'Peak Value', 'RMS', 'Shape Factor', 'Skewness', 'SD']
    cell_id_list = os.listdir(folder)
    print(cell_id_list)
    for cell_id in cell_id_list:
        files = os.listdir(folder+'/'+cell_id)
        feature_list = []
        for f in files:
            v_t = np.load(folder+'/'+cell_id+'/'+f)
            start = np.random.randint(0, len(v_t)-3600, size=1)[0]
            v_t = v_t[start:start+3600, 0]
            ewt, mfb ,boundaries = EWT1D(v_t, N=9)
            v_c = v_t-ewt[:, 0] 
            feature_list.append(feature_extraction(v_c))
        feature_list = np.vstack(feature_list)
        fig, ax = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            ax[i//5,i%5].plot(feature_list[:, i], lw=1)
            ax[i//5,i%5].set_title(feature_names[i], fontsize=8)
            ax[i//5,i%5].tick_params(axis='x', labelsize=5)
            ax[i//5,i%5].tick_params(axis='y', labelsize=5)
            ax[i//5,i%5].yaxis.get_offset_text().set(size=5)
        plt.subplots_adjust(wspace=0.5, hspace=0.2)
        print(cell_id)
        plt.show()
        plt.close()


def cycle_info_visualization(folder, cell_id, mode='full'):
    files = os.listdir(folder+'/'+cell_id)
    cmap = plt.get_cmap('coolwarm')
    for i, f in enumerate(files):
        cycle_info = np.load(folder+'/'+cell_id+'/'+f)
        if mode=='full':
            plt.plot(cycle_info[:, 1], c=cmap(i/len(files)), alpha=0.7, lw=0.5, label='current')
        elif mode=='partial':
            fig, ax1 = plt.subplots()
            plt.xlabel('Cycles')
            ax2 = ax1.twinx()
            ax1.plot(cycle_info[:3600, 0], c='blue', alpha=0.7, lw=0.5, label='voltage')
            ax2.plot(cycle_info[:3600, 1], c='red', alpha=0.7, lw=0.5, label='current')
            ax1.set_ylabel('Voltage (V)')
            ax2.set_ylabel('Current (I)')
            plt.legend()
            plt.show()
            plt.close()
        # plt.savefig('figs/%d.png'%(i+1))
    if mode=='full':
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
    # for cell in CELL_IDS:
    #     cycle_info_visualization('Stanford_Dataset/discharge_info', cell)
    npy_filtering('Stanford_Dataset')
    # ewt_visualization()
    # npy_to_selected_features('Stanford_Dataset/discharge_info')


if __name__ == '__main__':
    main()


