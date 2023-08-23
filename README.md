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
2. 傅立葉頻譜將會被分割為 $N$ 個modes
3. EWT的mode數量 $N$ 會跟傅立葉頻譜中的local maximum數量 $K$ 做比較。若 $K>N$，就會從所有local maximum中挑選 $N$ 個做後續的分析；若 $K&ltN$ ，則需要重新指定 $N$ 到適當的數值

![特徵選擇](https://hackmd.io/_uploads/ByR7Cil6h.png)

