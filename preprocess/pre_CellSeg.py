import os
import cv2
import glob
import tifffile as tif
import numpy as np
from skimage import measure, morphology, segmentation
from skimage.segmentation import slic,mark_boundaries
import random
import math
import json
from scipy.spatial import Voronoi, voronoi_plot_2d


brightfield_1_list = ['cell_00001.png', 'cell_00002.png', 'cell_00003.png', 'cell_00004.png', 'cell_00005.png', 'cell_00006.png', 'cell_00007.png', 'cell_00008.png', 'cell_00009.png', 'cell_00010.png', 'cell_00011.png', 'cell_00012.png', 'cell_00015.png', 'cell_00016.png', 'cell_00017.png', 'cell_00018.png', 'cell_00019.png', 'cell_00020.png', 'cell_00021.png', 'cell_00022.png', 'cell_00023.png', 'cell_00024.png', 'cell_00025.png', 'cell_00026.png', 'cell_00027.png', 'cell_00028.png', 'cell_00029.png', 'cell_00030.png', 'cell_00031.png', 'cell_00032.png', 'cell_00033.png', 'cell_00034.png', 'cell_00035.png', 'cell_00036.png', 'cell_00037.png', 'cell_00038.png', 'cell_00039.png', 'cell_00040.png', 'cell_00041.png', 'cell_00042.png', 'cell_00043.png', 'cell_00044.png', 'cell_00045.png', 'cell_00046.png', 'cell_00047.png', 'cell_00048.png', 'cell_00049.png', 'cell_00050.png', 'cell_00051.png', 'cell_00052.png', 'cell_00053.png', 'cell_00054.png', 'cell_00055.png', 'cell_00056.png', 'cell_00057.png', 'cell_00058.png', 'cell_00059.png', 'cell_00060.png', 'cell_00061.png', 'cell_00062.png', 'cell_00063.png', 'cell_00064.png', 'cell_00065.png', 'cell_00066.png', 'cell_00067.png', 'cell_00068.png', 'cell_00069.png', 'cell_00070.png', 'cell_00071.png', 'cell_00072.png', 'cell_00073.png', 'cell_00074.png', 'cell_00075.png', 'cell_00076.png', 'cell_00077.png', 'cell_00078.png', 'cell_00079.png', 'cell_00080.png', 'cell_00081.png', 'cell_00082.png', 'cell_00083.png', 'cell_00084.png', 'cell_00085.png', 'cell_00086.png', 'cell_00087.png', 'cell_00088.png', 'cell_00089.png', 'cell_00090.png', 'cell_00091.png', 'cell_00092.png', 'cell_00093.png', 'cell_00094.png', 'cell_00095.png', 'cell_00096.png', 'cell_00097.png', 'cell_00098.png', 'cell_00099.png', 'cell_00100.png', 'cell_00101.png', 'cell_00102.png', 'cell_00103.png', 'cell_00104.png', 'cell_00105.png', 'cell_00106.png', 'cell_00107.png', 'cell_00108.png', 'cell_00109.png', 'cell_00110.png', 'cell_00111.png', 'cell_00112.png', 'cell_00113.png', 'cell_00114.png', 'cell_00115.png', 'cell_00116.png', 'cell_00117.png', 'cell_00118.png', 'cell_00119.png', 'cell_00120.png', 'cell_00121.png', 'cell_00122.png', 'cell_00123.png', 'cell_00124.png', 'cell_00125.png', 'cell_00126.png', 'cell_00127.png', 'cell_00128.png', 'cell_00129.png', 'cell_00130.png', 'cell_00131.png', 'cell_00132.png', 'cell_00133.png', 'cell_00134.png', 'cell_00135.png', 'cell_00136.png', 'cell_00137.png', 'cell_00138.png', 'cell_00139.png', 'cell_00140.png', 'cell_00141.png']

brightfield_2_list = ['cell_00145.png', 'cell_00146.png', 'cell_00147.png', 'cell_00148.png', 'cell_00149.png', 'cell_00150.png', 'cell_00151.png', 'cell_00152.png', 'cell_00153.png', 'cell_00154.png', 'cell_00155.png', 'cell_00156.png', 'cell_00157.png', 'cell_00158.png', 'cell_00159.png', 'cell_00160.png', 'cell_00161.png', 'cell_00162.png', 'cell_00163.png', 'cell_00164.png', 'cell_00165.png', 'cell_00166.png', 'cell_00167.png', 'cell_00168.png', 'cell_00169.png', 'cell_00170.png', 'cell_00171.png', 'cell_00172.png', 'cell_00173.png', 'cell_00174.png', 'cell_00175.png', 'cell_00176.png', 'cell_00177.png', 'cell_00178.png', 'cell_00179.png', 'cell_00180.png', 'cell_00181.png', 'cell_00182.png', 'cell_00183.png', 'cell_00184.png', 'cell_00185.png', 'cell_00186.png', 'cell_00187.png', 'cell_00188.png', 'cell_00189.png', 'cell_00190.png', 'cell_00191.png', 'cell_00192.png', 'cell_00193.png', 'cell_00194.png', 'cell_00195.png', 'cell_00196.png', 'cell_00197.png', 'cell_00198.png', 'cell_00199.png', 'cell_00200.png', 'cell_00201.png', 'cell_00202.png', 'cell_00203.png', 'cell_00204.png', 'cell_00205.png', 'cell_00206.png', 'cell_00207.png', 'cell_00208.png', 'cell_00209.png', 'cell_00210.png', 'cell_00211.png', 'cell_00212.png', 'cell_00213.png', 'cell_00214.png', 'cell_00215.png', 'cell_00216.png', 'cell_00217.png', 'cell_00218.png', 'cell_00219.png', 'cell_00220.png', 'cell_00221.png', 'cell_00222.png', 'cell_00223.png', 'cell_00224.png', 'cell_00225.png', 'cell_00226.png', 'cell_00227.png', 'cell_00228.png', 'cell_00229.png', 'cell_00230.png']

DIC_list = ['cell_00501.png', 'cell_00502.png', 'cell_00503.png', 'cell_00504.png', 'cell_00505.png', 'cell_00506.png', 'cell_00507.png', 'cell_00508.png', 'cell_00509.png', 'cell_00510.png', 'cell_00511.png', 'cell_00512.png', 'cell_00513.png', 'cell_00514.png', 'cell_00515.png', 'cell_00516.png', 'cell_00517.png', 'cell_00518.png', 'cell_00519.png', 'cell_00520.png', 'cell_00521.png', 'cell_00522.png', 'cell_00523.png', 'cell_00524.png', 'cell_00525.png', 'cell_00528.png', 'cell_00529.png', 'cell_00530.png', 'cell_00531.png', 'cell_00532.png', 'cell_00533.png', 'cell_00534.png', 'cell_00535.png', 'cell_00536.png', 'cell_00537.png', 'cell_00538.png', 'cell_00539.png', 'cell_00540.png', 'cell_00541.png', 'cell_00542.png', 'cell_00543.png', 'cell_00544.png', 'cell_00557.png', 'cell_00558.png', 'cell_00559.png', 'cell_00560.png', 'cell_00561.png', 'cell_00562.png', 'cell_00563.png', 'cell_00564.png', 'cell_00565.png', 'cell_00566.png', 'cell_00567.png', 'cell_00568.png', 'cell_00569.png', 'cell_00570.png', 'cell_00571.png', 'cell_00572.png', 'cell_00573.png', 'cell_00574.png', 'cell_00575.png', 'cell_00576.png', 'cell_00577.png', 'cell_00578.png', 'cell_00579.png', 'cell_00580.png', 'cell_00581.png', 'cell_00582.png', 'cell_00583.png', 'cell_00584.png', 'cell_00585.png', 'cell_00586.png', 'cell_00587.png', 'cell_00588.png', 'cell_00589.png', 'cell_00590.png', 'cell_00591.png', 'cell_00592.png', 'cell_00593.png', 'cell_00594.png', 'cell_00595.png', 'cell_00596.png', 'cell_00597.png', 'cell_00598.png', 'cell_00599.png', 'cell_00600.png', 'cell_00601.png', 'cell_00602.png', 'cell_00603.png', 'cell_00604.png', 'cell_00605.png', 'cell_00606.png', 'cell_00607.png', 'cell_00608.png', 'cell_00609.png', 'cell_00610.png', 'cell_00611.png', 'cell_00612.png', 'cell_00613.png', 'cell_00614.png', 'cell_00615.png', 'cell_00616.png', 'cell_00617.png', 'cell_00618.png', 'cell_00619.png', 'cell_00621.png', 'cell_00622.png', 'cell_00623.png', 'cell_00624.png', 'cell_00625.png', 'cell_00626.png', 'cell_00627.png', 'cell_00628.png', 'cell_00629.png', 'cell_00630.png', 'cell_00631.png', 'cell_00632.png', 'cell_00633.png', 'cell_00634.png', 'cell_00635.png', 'cell_00636.png', 'cell_00637.png', 'cell_00638.png', 'cell_00639.png', 'cell_00640.png', 'cell_00641.png', 'cell_00642.png', 'cell_00643.png', 'cell_00644.png', 'cell_00645.png', 'cell_00646.png', 'cell_00647.png', 'cell_00648.png', 'cell_00649.png', 'cell_00650.png', 'cell_00651.png', 'cell_00652.png', 'cell_00653.png', 'cell_00654.png', 'cell_00655.png', 'cell_00656.png', 'cell_00657.png', 'cell_00658.png', 'cell_00659.png', 'cell_00660.png', 'cell_00661.png', 'cell_00662.png', 'cell_00663.png', 'cell_00664.png', 'cell_00665.png', 'cell_00666.png', 'cell_00667.png', 'cell_00668.png', 'cell_00669.png', 'cell_00670.png', 'cell_00671.png', 'cell_00672.png', 'cell_00673.png', 'cell_00674.png', 'cell_00675.png', 'cell_00676.png', 'cell_00677.png', 'cell_00692.png', 'cell_00693.png', 'cell_00694.png', 'cell_00695.png', 'cell_00696.png', 'cell_00697.png', 'cell_00698.png', 'cell_00699.png', 'cell_00700.png', 'cell_00701.png', 'cell_00702.png', 'cell_00703.png', 'cell_00704.png', 'cell_00705.png', 'cell_00706.png', 'cell_00707.png', 'cell_00708.png', 'cell_00709.png', 'cell_00710.png', 'cell_00711.png', 'cell_00712.png', 'cell_00713.png', 'cell_00714.png', 'cell_00715.png', 'cell_00716.png', 'cell_00717.png', 'cell_00718.png']

fluorescent_list = ['cell_00719.png', 'cell_00720.png', 'cell_00721.png', 'cell_00722.png', 'cell_00723.png', 'cell_00724.png', 'cell_00725.png', 'cell_00726.png', 'cell_00727.png', 'cell_00728.png', 'cell_00729.png', 'cell_00730.png', 'cell_00731.png', 'cell_00732.png', 'cell_00733.png', 'cell_00734.png', 'cell_00735.png', 'cell_00736.png', 'cell_00737.png', 'cell_00738.png', 'cell_00739.png', 'cell_00740.png', 'cell_00741.png', 'cell_00742.png', 'cell_00743.png', 'cell_00744.png', 'cell_00745.png', 'cell_00746.png', 'cell_00747.png', 'cell_00748.png', 'cell_00749.png', 'cell_00750.png', 'cell_00751.png', 'cell_00752.png', 'cell_00753.png', 'cell_00754.png', 'cell_00755.png', 'cell_00756.png', 'cell_00757.png', 'cell_00758.png', 'cell_00759.png', 'cell_00760.png', 'cell_00761.png', 'cell_00762.png', 'cell_00763.png', 'cell_00764.png', 'cell_00765.png', 'cell_00766.png', 'cell_00767.png', 'cell_00768.png', 'cell_00769.png', 'cell_00770.png', 'cell_00771.png', 'cell_00772.png', 'cell_00773.png', 'cell_00774.png', 'cell_00775.png', 'cell_00776.png', 'cell_00777.png', 'cell_00778.png', 'cell_00779.png', 'cell_00780.png', 'cell_00781.png', 'cell_00782.png', 'cell_00783.png', 'cell_00784.png', 'cell_00785.png', 'cell_00786.png', 'cell_00787.png', 'cell_00788.png', 'cell_00789.png', 'cell_00790.png', 'cell_00791.png', 'cell_00792.png', 'cell_00793.png', 'cell_00794.png', 'cell_00795.png', 'cell_00796.png', 'cell_00797.png', 'cell_00798.png', 'cell_00799.png', 'cell_00800.png', 'cell_00801.png', 'cell_00802.png', 'cell_00803.png', 'cell_00804.png', 'cell_00805.png', 'cell_00806.png', 'cell_00807.png', 'cell_00808.png', 'cell_00809.png', 'cell_00810.png', 'cell_00811.png', 'cell_00812.png', 'cell_00813.png', 'cell_00814.png', 'cell_00815.png', 'cell_00816.png', 'cell_00817.png', 'cell_00818.png', 'cell_00819.png', 'cell_00820.png', 'cell_00821.png', 'cell_00822.png', 'cell_00823.png', 'cell_00824.png', 'cell_00825.png', 'cell_00826.png', 'cell_00827.png', 'cell_00828.png', 'cell_00829.png', 'cell_00830.png', 'cell_00831.png', 'cell_00832.png', 'cell_00833.png', 'cell_00834.png', 'cell_00835.png', 'cell_00836.png', 'cell_00837.png', 'cell_00838.png', 'cell_00839.png', 'cell_00840.png', 'cell_00841.png', 'cell_00842.png', 'cell_00843.png', 'cell_00844.png', 'cell_00845.png', 'cell_00846.png', 'cell_00847.png', 'cell_00848.png', 'cell_00849.png', 'cell_00850.png', 'cell_00851.png', 'cell_00852.png', 'cell_00853.png', 'cell_00854.png', 'cell_00855.png', 'cell_00856.png', 'cell_00857.png', 'cell_00858.png', 'cell_00859.png', 'cell_00860.png', 'cell_00861.png', 'cell_00862.png', 'cell_00863.png', 'cell_00864.png', 'cell_00865.png', 'cell_00866.png', 'cell_00867.png', 'cell_00868.png', 'cell_00869.png', 'cell_00870.png', 'cell_00871.png', 'cell_00872.png', 'cell_00873.png', 'cell_00874.png', 'cell_00875.png', 'cell_00876.png', 'cell_00877.png', 'cell_00878.png', 'cell_00879.png', 'cell_00880.png', 'cell_00881.png', 'cell_00882.png', 'cell_00883.png', 'cell_00884.png', 'cell_00885.png', 'cell_00886.png', 'cell_00887.png', 'cell_00888.png', 'cell_00889.png', 'cell_00890.png', 'cell_00891.png', 'cell_00892.png', 'cell_00893.png', 'cell_00894.png', 'cell_00895.png', 'cell_00896.png', 'cell_00897.png', 'cell_00898.png', 'cell_00899.png', 'cell_00900.png', 'cell_00901.png', 'cell_00902.png', 'cell_00903.png', 'cell_00904.png', 'cell_00905.png', 'cell_00906.png', 'cell_00907.png', 'cell_00908.png', 'cell_00909.png', 'cell_00910.png', 'cell_00911.png', 'cell_00912.png', 'cell_00913.png', 'cell_00914.png', 'cell_00915.png', 'cell_00916.png', 'cell_00917.png', 'cell_00918.png', 'cell_00919.png', 'cell_00920.png', 'cell_00921.png', 'cell_00922.png', 'cell_00923.png', 'cell_00924.png', 'cell_00925.png', 'cell_00926.png', 'cell_00927.png', 'cell_00928.png', 'cell_00929.png', 'cell_00930.png', 'cell_00931.png', 'cell_00932.png', 'cell_00933.png', 'cell_00934.png', 'cell_00935.png', 'cell_00936.png', 'cell_00937.png', 'cell_00938.png', 'cell_00939.png', 'cell_00940.png', 'cell_00941.png', 'cell_00942.png', 'cell_00943.png', 'cell_00944.png', 'cell_00945.png', 'cell_00946.png', 'cell_00947.png', 'cell_00948.png', 'cell_00949.png', 'cell_00950.png', 'cell_00951.png', 'cell_00952.png', 'cell_00953.png', 'cell_00954.png', 'cell_00955.png', 'cell_00956.png', 'cell_00957.png', 'cell_00958.png', 'cell_00959.png', 'cell_00960.png', 'cell_00961.png', 'cell_00962.png', 'cell_00963.png', 'cell_00964.png', 'cell_00965.png', 'cell_00966.png', 'cell_00967.png', 'cell_00968.png', 'cell_00969.png', 'cell_00970.png', 'cell_00971.png', 'cell_00972.png', 'cell_00973.png', 'cell_00974.png', 'cell_00975.png', 'cell_00976.png', 'cell_00977.png', 'cell_00978.png', 'cell_00979.png', 'cell_00980.png', 'cell_00981.png', 'cell_00982.png', 'cell_00983.png', 'cell_00984.png', 'cell_00985.png', 'cell_00986.png', 'cell_00987.png', 'cell_00988.png', 'cell_00989.png', 'cell_00990.png', 'cell_00991.png', 'cell_00992.png', 'cell_00993.png', 'cell_00994.png', 'cell_00995.png', 'cell_00996.png', 'cell_00997.png', 'cell_00998.png', 'cell_00999.png', 'cell_01000.png']

PC_list = ['cell_00013.png', 'cell_00014.png', 'cell_00231.png', 'cell_00232.png', 'cell_00233.png', 'cell_00234.png', 'cell_00235.png', 'cell_00236.png', 'cell_00237.png', 'cell_00238.png', 'cell_00239.png', 'cell_00240.png', 'cell_00241.png', 'cell_00242.png', 'cell_00243.png', 'cell_00244.png', 'cell_00245.png', 'cell_00246.png', 'cell_00247.png', 'cell_00248.png', 'cell_00249.png', 'cell_00250.png', 'cell_00251.png', 'cell_00252.png', 'cell_00253.png', 'cell_00254.png', 'cell_00255.png', 'cell_00256.png', 'cell_00257.png', 'cell_00258.png', 'cell_00259.png', 'cell_00260.png', 'cell_00261.png', 'cell_00262.png', 'cell_00263.png', 'cell_00264.png', 'cell_00265.png', 'cell_00266.png', 'cell_00267.png', 'cell_00268.png', 'cell_00269.png', 'cell_00270.png', 'cell_00271.png', 'cell_00272.png', 'cell_00273.png', 'cell_00274.png', 'cell_00275.png', 'cell_00276.png', 'cell_00277.png', 'cell_00278.png', 'cell_00279.png', 'cell_00280.png', 'cell_00281.png', 'cell_00282.png', 'cell_00283.png', 'cell_00284.png', 'cell_00285.png', 'cell_00286.png', 'cell_00287.png', 'cell_00288.png', 'cell_00289.png', 'cell_00290.png', 'cell_00291.png', 'cell_00292.png', 'cell_00293.png', 'cell_00294.png', 'cell_00295.png', 'cell_00296.png', 'cell_00297.png', 'cell_00298.png', 'cell_00299.png', 'cell_00300.png', 'cell_00301.png', 'cell_00302.png', 'cell_00303.png', 'cell_00304.png', 'cell_00305.png', 'cell_00306.png', 'cell_00307.png', 'cell_00308.png', 'cell_00309.png', 'cell_00310.png', 'cell_00311.png', 'cell_00312.png', 'cell_00313.png', 'cell_00314.png', 'cell_00315.png', 'cell_00316.png', 'cell_00317.png', 'cell_00318.png', 'cell_00319.png', 'cell_00320.png', 'cell_00321.png', 'cell_00322.png', 'cell_00323.png', 'cell_00324.png', 'cell_00325.png', 'cell_00326.png', 'cell_00327.png', 'cell_00328.png', 'cell_00329.png', 'cell_00330.png', 'cell_00331.png', 'cell_00332.png', 'cell_00333.png', 'cell_00334.png', 'cell_00335.png', 'cell_00336.png', 'cell_00337.png', 'cell_00338.png', 'cell_00339.png', 'cell_00340.png', 'cell_00341.png', 'cell_00342.png', 'cell_00343.png', 'cell_00344.png', 'cell_00345.png', 'cell_00346.png', 'cell_00347.png', 'cell_00348.png', 'cell_00349.png', 'cell_00350.png', 'cell_00351.png', 'cell_00352.png', 'cell_00353.png', 'cell_00354.png', 'cell_00355.png', 'cell_00356.png', 'cell_00357.png', 'cell_00358.png', 'cell_00359.png', 'cell_00360.png', 'cell_00361.png', 'cell_00362.png', 'cell_00363.png', 'cell_00364.png', 'cell_00365.png', 'cell_00366.png', 'cell_00367.png', 'cell_00368.png', 'cell_00369.png', 'cell_00370.png', 'cell_00371.png', 'cell_00372.png', 'cell_00373.png', 'cell_00374.png', 'cell_00375.png', 'cell_00376.png', 'cell_00377.png', 'cell_00378.png', 'cell_00379.png', 'cell_00380.png', 'cell_00381.png', 'cell_00382.png', 'cell_00383.png', 'cell_00384.png', 'cell_00385.png', 'cell_00386.png', 'cell_00387.png', 'cell_00388.png', 'cell_00389.png', 'cell_00390.png', 'cell_00391.png', 'cell_00392.png', 'cell_00393.png', 'cell_00394.png', 'cell_00395.png', 'cell_00396.png', 'cell_00397.png', 'cell_00398.png', 'cell_00399.png', 'cell_00400.png', 'cell_00401.png', 'cell_00402.png', 'cell_00403.png', 'cell_00404.png', 'cell_00405.png', 'cell_00406.png', 'cell_00407.png', 'cell_00408.png', 'cell_00409.png', 'cell_00410.png', 'cell_00411.png', 'cell_00412.png', 'cell_00413.png', 'cell_00414.png', 'cell_00415.png', 'cell_00416.png', 'cell_00417.png', 'cell_00418.png', 'cell_00419.png', 'cell_00420.png', 'cell_00421.png', 'cell_00422.png', 'cell_00423.png', 'cell_00424.png', 'cell_00425.png', 'cell_00426.png', 'cell_00427.png', 'cell_00428.png', 'cell_00429.png', 'cell_00430.png', 'cell_00431.png', 'cell_00432.png', 'cell_00433.png', 'cell_00434.png', 'cell_00435.png', 'cell_00436.png', 'cell_00437.png', 'cell_00438.png', 'cell_00439.png', 'cell_00440.png', 'cell_00441.png', 'cell_00442.png', 'cell_00443.png', 'cell_00444.png', 'cell_00445.png', 'cell_00446.png', 'cell_00447.png', 'cell_00448.png', 'cell_00449.png', 'cell_00450.png', 'cell_00451.png', 'cell_00452.png', 'cell_00453.png', 'cell_00454.png', 'cell_00455.png', 'cell_00456.png', 'cell_00457.png', 'cell_00458.png', 'cell_00459.png', 'cell_00460.png', 'cell_00461.png', 'cell_00462.png', 'cell_00463.png', 'cell_00464.png', 'cell_00465.png', 'cell_00466.png', 'cell_00467.png', 'cell_00468.png', 'cell_00469.png', 'cell_00470.png', 'cell_00471.png', 'cell_00472.png', 'cell_00473.png', 'cell_00474.png', 'cell_00475.png', 'cell_00476.png', 'cell_00477.png', 'cell_00478.png', 'cell_00479.png', 'cell_00480.png', 'cell_00481.png', 'cell_00482.png', 'cell_00483.png', 'cell_00484.png', 'cell_00485.png', 'cell_00486.png', 'cell_00487.png', 'cell_00488.png', 'cell_00489.png', 'cell_00490.png', 'cell_00491.png', 'cell_00492.png', 'cell_00493.png', 'cell_00494.png', 'cell_00495.png', 'cell_00496.png', 'cell_00497.png', 'cell_00498.png', 'cell_00499.png', 'cell_00500.png', 'cell_00526.png', 'cell_00527.png', 'cell_00546.png', 'cell_00547.png', 'cell_00548.png', 'cell_00549.png', 'cell_00550.png', 'cell_00551.png', 'cell_00552.png', 'cell_00553.png', 'cell_00554.png', 'cell_00555.png', 'cell_00556.png']

other_list = ['cell_00142.png', 'cell_00143.png', 'cell_00144.png', 'cell_00545.png', 'cell_00620.png', 'cell_00678.png', 'cell_00679.png', 'cell_00680.png', 'cell_00681.png', 'cell_00682.png', 'cell_00683.png', 'cell_00684.png', 'cell_00685.png', 'cell_00686.png', 'cell_00687.png', 'cell_00688.png', 'cell_00689.png', 'cell_00690.png', 'cell_00691.png']

def gen_point_supervision(semantic):
    point_dict = {"background":{}, "foreground":{}}
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(semantic.astype(np.uint8),connectivity=8)
    # select the foreground points
    num_cells = np.max(labels)
    for ii in range(1,num_cells+1):
        x,y,w,h,s = stats[ii]
        centroid = centroids[ii]
        centroid = list(map(int,centroid))
            
        flag = 0
        attempt_num = 0
        point_range = 0.1
        while flag == 0:
            if attempt_num == 20:
                point_range += 0.05
                attempt_num = 0
            offset_ratio_x = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_ratio_y = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_x = round(w*offset_ratio_x)
            offset_y = round(h*offset_ratio_y)
            point_x = centroid[0] + offset_x
            point_y = centroid[1] + offset_y
            if labels[point_y, point_x] == ii:
                flag = 1
            else:
                attempt_num += 1
            if point_range > 0.3:
                print("False point in image!!")
        
        point_dict['foreground'][str(ii)] = {'x':int(x),
                                            'y':int(y),
                                            'w':int(w),
                                            'h':int(h),
                                            's':int(s),
                                            'centroid':centroid,
                                            "select_point":[point_x, point_y]}
    # select the background points
    background = np.zeros(semantic.shape)
    H,W = background.shape
    background[semantic==1] = 1
    background_points= []
    need_point_num = max(10,round(num_cells))
    while len(background_points)<need_point_num:
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        if background[y][x] == 0:
            background_points.append([x,y])
    for ii, point in enumerate(background_points):
        point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                'y':int(point[1])}
    return point_dict

def get_distance(pt1, pt2):
    dist = np.sqrt(np.power(pt1[0]-pt2[0],2)+np.power(pt1[1]-pt2[1],2))
    return dist

def get_min_distance(fg_point_dict):
    select_points = [v['select_point'] for k,v in fg_point_dict.items()]
    if len(select_points)>=2:
        min_dist_list = []
        for i, pt1 in enumerate(select_points):
            dist_list = []
            for ii, pt2 in enumerate(select_points):
                dist_list.append(get_distance(pt1, pt2))
            dist_list.remove(0)
            min_dist_list.append(min(dist_list))
        min_dist = min(min_dist_list)
    else:
        min_dist = 100
    min_dist = min(100,min_dist)
    return min_dist

def gen_superpixel_adaptive(img, point_dict):
    H,W,C = img.shape
    # find the minimum distance between the foreground
    fg_points_dict = point_dict['foreground']
    min_dist = get_min_distance(fg_points_dict)
    min_dist = max(20, min_dist)
    h_num = H/min_dist
    w_num = W/min_dist
    n_segment = h_num*w_num*2
    seg = slic(img, n_segments=n_segment, compactness=10, sigma=3, max_num_iter=5, slic_zero=True)
    vis = mark_boundaries(img,seg)*255
    return seg, vis

def gen_weak_label_with_point_fb(img, point_dict, superpixel, **kwargs):
    weak_label = np.ones(img.shape[:2])*255
    weak_label_count = np.zeros(img.shape[:2])
    weak_label_valid = np.zeros(img.shape[:2])
    fg_points_dict = point_dict['foreground']
    bg_points_dict = point_dict['background']
    instance_dict = {"superpixels":{}, "instances":{}, 'background':[], 'labeled_sp':[]}
    for superpixel_idx in range(1,np.max(superpixel)+1):
        instance_dict['superpixels'][superpixel_idx]=-1
        
    # For background
    for point_idx, bg_point_prop in bg_points_dict.items():
        point_x = bg_point_prop['x']
        point_y = bg_point_prop['y']
        superpixel_value = superpixel[point_y, point_x]
        # get the valid region
        weak_label_valid[superpixel==superpixel_value] = 1
        # get the labeled superpixel boundary 
        weak_temp = np.zeros(img.shape[:2], dtype=np.uint8)
        weak_temp[superpixel == superpixel_value] = 1
        weak_temp_boundary = segmentation.find_boundaries(weak_temp, mode='outer')
        # set the weak label
        weak_label[superpixel == superpixel_value] = 0
        # get the counting image for adj pixels
        weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
        instance_dict['superpixels'][str(superpixel_value)] = 0
        instance_dict['background'].append(int(superpixel_value))
        instance_dict['labeled_sp'].append(int(superpixel_value))
    
    # For foreground
    for point_idx, fg_point_prop in fg_points_dict.items():
        point_x, point_y = fg_point_prop['select_point']
        superpixel_value = superpixel[point_y,point_x]
        # get the valid region
        weak_label_valid[superpixel==superpixel_value] = 1
        # get the labeled superpixel boundary 
        weak_temp = np.zeros(img.shape[:2], dtype=np.uint8)
        weak_temp[superpixel == superpixel_value] = 1
        weak_temp_boundary = segmentation.find_boundaries(weak_temp, mode='outer')
        # set the weak label
        weak_label[superpixel == superpixel_value] = 1
        # get the counting image for adj pixels
        weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
        instance_dict['superpixels'][str(superpixel_value)] = int(point_idx)
        instance_dict['instances'][str(point_idx)] = [int(superpixel_value)]
        instance_dict['labeled_sp'].append(int(superpixel_value))
    
    # fine-grained fix: 
    adj_pixels = np.zeros(img.shape[:2])
    adj_pixels[weak_label_count>1] = 1
    adj_pixels = morphology.dilation(adj_pixels, morphology.disk(1))
    adj_pixels = adj_pixels*weak_label_valid
    weak_label[adj_pixels == 1] = 0
    
    return weak_label, instance_dict, weak_label_valid

def gen_full_label_with_semantic(semantic):
    full_label = np.zeros(semantic.shape)
    full_label[semantic>0]=1
    return full_label

def create_interior_map(inst_map, disk_size):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.
    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(disk_size))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 0          # it used to be 2
    return interior

def preprocess_CellSeg(image_dir, gt_dir, output_dir):
    print("Preprocessing the CellSeg dataset...")
    output_semantic_dir = os.path.join(output_dir, 'semantics')
    output_vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(output_semantic_dir,exist_ok=True)
    os.makedirs(output_vis_dir,exist_ok=True)
    for i, image_name in enumerate(os.listdir(image_dir)):
        print("Processing the {}/{} images...".format(i, len(os.listdir(image_dir))), end='\r')
        image_path = os.path.join(image_dir, image_name)
        gt_path = os.path.join(gt_dir, image_name.replace('.png', '_label.tiff'))
        img = cv2.imread(image_path)
        gt = tif.imread(gt_path)
        H,W,C = img.shape
        
        semantic = create_interior_map(gt)
        semantic[semantic == 1] = 128
        save_semantic_path = os.path.join(output_semantic_dir, image_name)
        cv2.imwrite(save_semantic_path, semantic)

        vis = np.zeros_like(gt)
        vis[gt>0] = 255
        vis = vis[:,:,np.newaxis]
        vis = np.concatenate((vis,vis,vis), axis=2)
        vis_path = os.path.join(output_vis_dir,image_name)
        vis = np.hstack((img, vis))
        cv2.imwrite(vis_path, vis)
    pass

def gen_full_n_weak_heatmap(full_label, weak_label, weak_label_valid, disk_size):
    full_label = full_label.astype(np.float32)
    full_heat = cv2.GaussianBlur(full_label, (disk_size,disk_size), -1)*255
    
    weak_label[weak_label==255] = 0
    weak_label = morphology.dilation(weak_label, morphology.disk(disk_size-1))
    weak_heat = cv2.GaussianBlur(weak_label.astype(np.float32), (disk_size, disk_size), -1)
    weak_heat = weak_heat*weak_label_valid
    weak_heat = weak_heat/np.max(weak_heat)*255
    return full_heat, weak_heat

def gen_weak_n_full_labels(image_dir, semantic_dir, output_dir, image_list, **kwargs):
    print("Generating the weak and full supervision labels...")
    output_superpixel_dir = os.path.join(output_dir,'superpixels')
    output_point_dir = os.path.join(output_dir,'points')
    output_weak_dir = os.path.join(output_dir, 'labels/weak')
    output_weak_heat_dir = os.path.join(output_dir, 'labels/weak_heatmap')
    output_inst_dir = os.path.join(output_dir, 'labels/weak_inst')
    output_full_dir = os.path.join(output_dir, 'labels/full')
    output_full_heat_dir = os.path.join(output_dir, 'labels/full_heatmap')
    os.makedirs(output_superpixel_dir, exist_ok=True)
    os.makedirs(output_point_dir, exist_ok=True)
    os.makedirs(output_weak_dir, exist_ok=True)
    os.makedirs(output_weak_heat_dir, exist_ok=True)
    os.makedirs(output_inst_dir, exist_ok=True)
    os.makedirs(output_full_dir, exist_ok=True)
    os.makedirs(output_full_heat_dir, exist_ok=True)
    for ii, image_name in enumerate(image_list):
        print("Generating the {}/{} supervision labels...".format(ii, len(image_list)), end='\r')
        img = cv2.imread(os.path.join(image_dir, image_name))
        semantic = cv2.imread(os.path.join(semantic_dir, image_name), flags=0)
        semantic[semantic==128] = 1
        point_dict = gen_point_supervision(semantic)
        superpixel, sp_vis = gen_superpixel_adaptive(img, point_dict)
        weak_label, instance_dict, weak_label_valid = gen_weak_label_with_point_fb(img, point_dict, superpixel, **kwargs)
        full_label = gen_full_label_with_semantic(semantic)
        full_heat, weak_heat = gen_full_n_weak_heatmap(full_label.copy(), weak_label.copy(), weak_label_valid.copy(), **kwargs)
        # save the mid-results
        json.dump(point_dict, open(os.path.join(output_point_dir, image_name.replace('.png','.json')), 'w'),indent=2)
        tif.imwrite(os.path.join(output_superpixel_dir, image_name.replace('.png', '.tiff')), superpixel)
        cv2.imwrite(os.path.join(output_superpixel_dir, image_name.replace('.png', '_vis.png')), sp_vis)
        cv2.imwrite(os.path.join(output_weak_dir, image_name), weak_label)
        cv2.imwrite(os.path.join(output_weak_heat_dir, image_name), weak_heat)
        cv2.imwrite(os.path.join(output_full_dir, image_name), full_label)
        cv2.imwrite(os.path.join(output_full_heat_dir, image_name), full_heat)
        json.dump(instance_dict, open(os.path.join(output_inst_dir, image_name.replace('.png','.json')), 'w'), indent=2)
    pass

if __name__=="__main__":
    image_dir = './data/NeurIPS2022_CellSeg/images'
    gt_dir = './data/NeurIPS2022_CellSeg/gts'
    semantic_dir = './data/NeurIPS2022_CellSeg/semantics'
    output_dir = './data/NeurIPS2022_CellSeg'
    kwargs = {"image_list": brightfield_1_list, "disk_size": 5}
    image_list = brightfield_1_list
    preprocess_CellSeg(image_dir, gt_dir,  output_dir)
    gen_weak_n_full_labels(image_dir, semantic_dir, output_dir, **kwargs)
    pass