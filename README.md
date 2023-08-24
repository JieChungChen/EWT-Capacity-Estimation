## Capacity estimation of lithium-ion batteries based on adaptive empirical wavelet transform and long short-term memory neural network  

[Capacity estimation of lithium-ion batteries based on adaptive empirical wavelet transform and long short-term memory neural network (Journal of Energy Storage 2023)](https://www.sciencedirect.com/science/article/pii/S2352152X23014433#t0020)

### 大綱  
* 利用經由Urban Dynamometer Driving Schedule (UDDS)生成的電池資料集，模擬鋰電池在電動車的使用情境  
* 透過Empirical Wavelet Transform (EWT)萃取電動車電池的放電資訊  
* 僅使用了量測資料中的電壓資訊作為特徵，減少了量測設備上的需求  

![流程](https://ars.els-cdn.com/content/image/1-s2.0-S2352152X23014433-ga1_lrg.jpg)  

### 資料集  

* NASA資料集(2014): 28顆LG Chem生產的LCO 18650電芯，標準容量為2.1Ah。放電時，每隔一段時間就會隨機調整電流。
* Stanford資料集: 10顆LG Chem生產的INR21700-M50T NMC電芯在23℃下，標準容量為4.85Ah。使用Urban Dynamometer Driving Schedule (UDDS)進行充放電實驗，模擬電芯在電動車的使用情境。每顆電芯會在約25個循環後做完整的reference performance test (RPT)

### Empirical Wavelet Transform (EWT)  

EWT是有自適應能力的高階訊號處理方法，可用於分解非線性、不平穩的訊號。EWT的實現步驟大致如下。

1. 透過傅立葉轉換將時域訊號 $v(t)$ 轉成頻譜 $\hat{v}(\omega)$
2. 傅立葉頻譜將會被分割為$N$個modes
3. EWT的mode數量 $N$ 會跟傅立葉頻譜中的local maximum數量 $K$ 做比較。若 $K \gt N$，就會從所有local maximum中保留數值最大的前 $N$ 個做後續的分析；若 $K \lt N$ ，則需要重新指定 $N$ 到適當的數值
4. 取相鄰最大值的中間頻率 $\lambda l,\ l=1,2,...N$ 作為邊界
5. 根據計算出的邊界，設計出 $N$ 個小波濾波器，由一個低通(low-pass)濾波器和 $N-1$ 個帶通(bandpass)濾波器組成

原始電壓訊號 $v(t)$ 經由EWT分解成9種mode後如下圖所示，其中(a)為NASA資料集的電芯(b)為Stanford資料集中的電芯  

![](https://ars.els-cdn.com/content/image/1-s2.0-S2352152X23014433-gr11_lrg.jpg)

### 基本模式過濾(Fundamental mode filtering)  

由於EWT會將電壓訊號分解為 $N$ 個mode，因此電壓訊號 $v_{n}(t)$ 可以表示成:  

$v_{n}(t)=\sum_{i=1}^{n} Mode_{i}(t)+Mode_{f}(t)+res(t)$  

其中 $Mode_{f}(t)$ 代表的是 $n$ 種mode中，跟原始電壓訊號趨勢最接近的基本模式。同時，這也是跟電芯老化與否最不相關的模式，必須透過基本模式過濾的步驟加以去除。首先透過相關係數計算出和原始訊號最高相關性的mode，也就是基本模式 $Mode_{f}(t)$ 。再將 $v_{n}(t)$ 扣除該模式，即可得到過濾後的電壓訊號 $v_{c}(t)$ ，計算過程如下:  

$v_{c}(t)=v_{n}(t)-Mode_{f}(t)$

### 特徵篩選  

透過基本模式過濾的電壓訊號 $v_{c}(t)$ 會以下表的方式計算出13項統計特徵  

![特徵選擇](https://hackmd.io/_uploads/ByR7Cil6h.png)
