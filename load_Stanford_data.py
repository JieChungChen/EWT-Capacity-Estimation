import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from ewtpy import EWT1D
CELL_IDS = ['G1', 'V4', 'V5', 'W4', 'W5', 'W8', 'W9', 'W10']
CYCLE_DICT = {'W3':[0,25,75],
              'W4':[0,25,75,123,132,159,176,179],
              'W5':[0,25,75,125,159,167,187,194,219,244,269,294,319,344,369],
              'W7':[0,25,75,125],
              'W8':[0,25,75,125,148,150,151,157,185,222,247,272,297,322,347],
              'W9':[0,25,75,122,144,145,146,150,179,216,241,266,291,316,341],
              'W10':[0,25,75,122,146,148,151,159,188,225,250,275,300,325,350],
              'G1':[0,25,30,37,62,87,112,137,162,187,212],
              'V4':[0,20,45,70,95,120,145,170,194,219,244],
              'V5':[0,12,18,29]}


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
    ignore_dict = {'G1':[], 'V4':[71], 'V5':[], 'W4':[], 'W5':range(76, 126, 1)+[195], 'W8':[], 'W9':[140, 170], 'W10':[179]}
    for cell_id in CELL_IDS:
        real_cycle = CYCLE_DICT[cell_id]
        count = 1
        accum_cycle = 1
        for cycle in range(14):
            filepath = '%s/Cycling_%d/%s.mat' % (folder, cycle+1, cell_id)
            if os.path.exists(filepath):
                v, c, s = get_mat_voltage(filepath)
                v[s!=14] = 0 # remove non-discharging part
                dif = np.abs(np.diff(v, axis=0, append=0)) # calculate discharge boundaries
                edge = np.argwhere(dif>1)+1 # find discharge boundaries
                dc_v=[v[edge[i, 0]:edge[i+1, 0]] for i in range(0, len(edge)-1, 2)] # separate discharge area, len(edge)-1 for odd number boundaries
                dc_c=[c[edge[i, 0]:edge[i+1, 0]] for i in range(0, len(edge)-1, 2)]

                for i in range(len(dc_v)):
                    # cycle_info shape: (time_step, 2)
                    v, c = dc_v[i], dc_c[i]
                    cycle_info = np.hstack([v, c])
                    if (i+accum_cycle) not in ignore_dict[cell_id]:
                        points = np.arange(0, len(cycle_info), sep)
                        cycle_info = cycle_info[points]
                        os.makedirs('%s/discharge_info/%s'%(folder, cell_id), exist_ok=True)
                        np.save('%s/discharge_info/%s/cycle_%s.npy'%(folder, cell_id, str(i+accum_cycle).zfill(3)), cycle_info)
                accum_cycle=real_cycle[count]+1
                count+=1
        print('%s finished'%(cell_id))

        
def cycle_info_visualization(folder, cell_id, mode='full'):
    files = os.listdir(folder+'/'+cell_id)
    cmap = plt.get_cmap('coolwarm')
    for i, f in enumerate(files):
        cycle_info = np.load(folder+'/'+cell_id+'/'+f)
        if mode=='full':
            plt.plot(cycle_info[:, 0], c=cmap(i/len(files)), alpha=0.7, lw=0.5, label='current')
        elif mode=='partial':
            fig, ax1 = plt.subplots()
            plt.xlabel('Time (s)', fontsize=14)
            ax1.plot(cycle_info[:, 0], c='blue', alpha=0.7, lw=0.5, label='voltage')
            ax1.set_ylabel('Voltage (V)', c='blue', fontsize=14)
            # ax2 = ax1.twinx()
            # ax2.plot(cycle_info[:, 1], c='red', alpha=0.7, lw=0.5, label='current')    
            # ax2.set_ylabel('Current (A)', c='red', fontsize=14)
            # plt.show()
            os.makedirs('figs/%s'%(cell_id), exist_ok=True)
            plt.savefig('figs/%s/%s.png'%(cell_id, f[:-4]))
            plt.close()
    if mode=='full':
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=1, vmax=len(files)),cmap='coolwarm')
        sm.set_array([])
        plt.colorbar(sm)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.show()
        plt.close()

    
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


def npy_filtering_sampler(folder, s_ratio=20):
    """
    for each cycle, use EWT to extract (s_ratio, 3600) array
    隨機從一個循環中挑s_ratio個片段, 並對每個片段做fundamental mode filtering
    """
    cell_id_list = os.listdir(folder+'/discharge_info')
    for cell_id in cell_id_list:
        os.makedirs('%s/filtered_info_vi/%s'%(folder, cell_id), exist_ok=True)
        files = os.listdir(folder+'/discharge_info/'+cell_id)
        for i, f in enumerate(files):
            features = np.load(folder+'/discharge_info/'+cell_id+'/'+f)
            v_c_list = []
            start_p = np.random.choice(len(features)-3600, s_ratio, replace=False)
            for p in start_p:
                seg = features[p:(p+3600), :].copy()
                ewt, mfb ,boundaries = EWT1D(seg[:, 0], N=9)
                seg[:, 0] = seg[:, 0]-ewt[:, 0]
                v_c_list.append(seg.transpose((1, 0)).reshape(1, 2, -1))
            # print(np.vstack(v_c_list).shape)
            np.save('%s/filtered_info_vi/%s/cycle_%s.npy'%(folder, cell_id, f[-7:-4]), np.vstack(v_c_list))


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
    f_mode = ewt[:, 0]
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


def capacity_test(filepath):
    f = loadmat(filepath)
    print(f.keys())
    capacity = f['cap']
    cells = f['col_cell_label']
    for i in range(10):
        c_list = []
        for c in capacity[:, i]:
            if not np.isnan(c[-1, 0]):
                c_list.append(c[-1, 0])
        cycle = CYCLE_DICT[cells[0, i][0]]
        # capacity_i = np.vstack([cycle, c_list])
        # print(capacity_i.shape)
        # np.save('Stanford_Dataset/capacity_each_cell/%s_capacity.npy'%(cells[0, i][0]), capacity_i)
        plt.plot(cycle, c_list, c='black', ls='--')
        plt.scatter(cycle, c_list, c='red')
        plt.title(cells[0, i][0], fontsize=14)
        plt.ylabel('capacity(Ah)')
        plt.savefig('Stanford_Dataset/capacity_each_cell/%s.png'%(cells[0, i][0]))
        plt.close()


def main():
    # load .mat files and save as .npy files 
    # mat_to_npy('Stanford_Dataset')

    # for cell in CELL_IDS:
    #     cycle_info_visualization('Stanford_Dataset/discharge_info', cell, mode='partial')

    # f mode filtering for each discharge segment
    npy_filtering_sampler('Stanford_Dataset')
    # ewt_visualization()
    # npy_to_selected_features('Stanford_Dataset/discharge_info')

    # load & save capacity test result
    # capacity_test('Stanford_Dataset/capacity_test.mat')


if __name__ == '__main__':
    main()


