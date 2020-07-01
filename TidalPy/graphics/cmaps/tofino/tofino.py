# 
#         tofino
#                   www.fabiocrameri.ch/colourmaps
from matplotlib.colors import LinearSegmentedColormap      
      
cm_data = [[0.87044, 0.84978, 0.99992],      
           [0.85983, 0.84232, 0.99532],      
           [0.84923, 0.83488, 0.99073],      
           [0.83862, 0.82744, 0.98614],      
           [0.82803, 0.82001, 0.98155],      
           [0.81744, 0.8126, 0.97696],      
           [0.80685, 0.80519, 0.97238],      
           [0.79627, 0.7978, 0.96779],      
           [0.7857, 0.79041, 0.9632],      
           [0.77513, 0.78303, 0.95862],      
           [0.76458, 0.77566, 0.95404],      
           [0.75403, 0.76829, 0.94946],      
           [0.74349, 0.76094, 0.94489],      
           [0.73296, 0.75358, 0.94031],      
           [0.72244, 0.74624, 0.93573],      
           [0.71193, 0.73891, 0.93116],      
           [0.70143, 0.73157, 0.92659],      
           [0.69094, 0.72424, 0.92202],      
           [0.68047, 0.71693, 0.91745],      
           [0.67001, 0.70961, 0.91288],      
           [0.65956, 0.70229, 0.90831],      
           [0.64912, 0.69498, 0.90373],      
           [0.63869, 0.68767, 0.89915],      
           [0.62827, 0.68036, 0.89457],      
           [0.61786, 0.67306, 0.88996],      
           [0.60747, 0.66575, 0.88535],      
           [0.59708, 0.65844, 0.88072],      
           [0.58669, 0.65113, 0.87608],      
           [0.57633, 0.6438, 0.8714],      
           [0.56595, 0.63648, 0.86669],      
           [0.55558, 0.62912, 0.86194],      
           [0.54523, 0.62176, 0.85713],      
           [0.53488, 0.61438, 0.85228],      
           [0.52453, 0.60699, 0.84735],      
           [0.51419, 0.59956, 0.84234],      
           [0.50383, 0.5921, 0.83724],      
           [0.49349, 0.58461, 0.83202],      
           [0.48315, 0.57708, 0.82669],      
           [0.47281, 0.5695, 0.82121],      
           [0.46249, 0.5619, 0.81559],      
           [0.4522, 0.55424, 0.80979],      
           [0.44191, 0.54653, 0.80381],      
           [0.43166, 0.53878, 0.79763],      
           [0.42143, 0.53097, 0.79123],      
           [0.41127, 0.52313, 0.78461],      
           [0.40119, 0.51523, 0.77775],      
           [0.39118, 0.5073, 0.77064],      
           [0.38127, 0.49933, 0.76327],      
           [0.37146, 0.49137, 0.75565],      
           [0.36183, 0.48335, 0.74777],      
           [0.35233, 0.47535, 0.73963],      
           [0.34304, 0.46735, 0.73123],      
           [0.33394, 0.45936, 0.72259],      
           [0.32504, 0.4514, 0.71373],      
           [0.31641, 0.44349, 0.70464],      
           [0.30802, 0.43563, 0.69535],      
           [0.29991, 0.42782, 0.68588],      
           [0.29207, 0.42009, 0.67624],      
           [0.28451, 0.41245, 0.66646],      
           [0.27726, 0.40487, 0.65655],      
           [0.27029, 0.39741, 0.64654],      
           [0.26359, 0.39004, 0.63644],      
           [0.25716, 0.38279, 0.62626],      
           [0.25101, 0.37562, 0.61604],      
           [0.24508, 0.36857, 0.60577],      
           [0.23943, 0.36161, 0.5955],      
           [0.23398, 0.35476, 0.58521],      
           [0.22876, 0.34798, 0.57491],      
           [0.22369, 0.34131, 0.56463],      
           [0.21881, 0.33473, 0.55436],      
           [0.21409, 0.3282, 0.54411],      
           [0.20949, 0.32177, 0.53389],      
           [0.20502, 0.31541, 0.52371],      
           [0.20064, 0.30912, 0.51356],      
           [0.19641, 0.30285, 0.50343],      
           [0.19221, 0.29668, 0.49335],      
           [0.18813, 0.29053, 0.48331],      
           [0.18406, 0.28444, 0.47331],      
           [0.18006, 0.27841, 0.46335],      
           [0.17611, 0.27238, 0.45343],      
           [0.17224, 0.26643, 0.44356],      
           [0.16839, 0.2605, 0.43371],      
           [0.16459, 0.25462, 0.42392],      
           [0.1608, 0.24876, 0.41418],      
           [0.15705, 0.2429, 0.40446],      
           [0.15332, 0.23714, 0.3948],      
           [0.14967, 0.23137, 0.38518],      
           [0.14601, 0.22564, 0.37561],      
           [0.14238, 0.21994, 0.36609],      
           [0.13878, 0.21428, 0.3566],      
           [0.13523, 0.20865, 0.34717],      
           [0.13171, 0.20306, 0.33777],      
           [0.12823, 0.19751, 0.32844],      
           [0.12476, 0.192, 0.31915],      
           [0.12134, 0.18655, 0.30992],      
           [0.11804, 0.18112, 0.30072],      
           [0.11468, 0.17575, 0.29162],      
           [0.11149, 0.17047, 0.28256],      
           [0.10825, 0.16519, 0.27356],      
           [0.10512, 0.16001, 0.26461],      
           [0.10208, 0.15492, 0.25575],      
           [0.099092, 0.14991, 0.24696],      
           [0.096147, 0.14496, 0.23825],      
           [0.093305, 0.14005, 0.22961],      
           [0.090594, 0.13531, 0.22111],      
           [0.087928, 0.13068, 0.21266],      
           [0.085292, 0.12614, 0.20436],      
           [0.082823, 0.12168, 0.19614],      
           [0.080421, 0.11744, 0.18808],      
           [0.078086, 0.11339, 0.18011],      
           [0.075905, 0.10947, 0.17234],      
           [0.073885, 0.10569, 0.16474],      
           [0.07186, 0.10213, 0.1573],      
           [0.069982, 0.098819, 0.15012],      
           [0.068226, 0.095733, 0.14311],      
           [0.066537, 0.092861, 0.13632],      
           [0.064955, 0.090276, 0.12989],      
           [0.063267, 0.087897, 0.12362],      
           [0.061329, 0.085713, 0.11774],      
           [0.059549, 0.084005, 0.11217],      
           [0.057509, 0.082538, 0.10693],      
           [0.055684, 0.081475, 0.10197],      
           [0.05386, 0.080759, 0.097418],      
           [0.05223, 0.080481, 0.093171],      
           [0.050913, 0.080625, 0.089271],      
           [0.049943, 0.081198, 0.085513],      
           [0.049353, 0.08218, 0.082189],      
           [0.049079, 0.083464, 0.079009],      
           [0.049155, 0.085141, 0.076274],      
           [0.049556, 0.087226, 0.074],      
           [0.050267, 0.089512, 0.072019],      
           [0.051322, 0.092107, 0.0705],      
           [0.052671, 0.094931, 0.069496],      
           [0.054312, 0.097914, 0.068834],      
           [0.05624, 0.10103, 0.068579],      
           [0.058195, 0.10432, 0.068741],      
           [0.06029, 0.10787, 0.06926],      
           [0.062176, 0.11172, 0.07],      
           [0.063928, 0.11578, 0.071095],      
           [0.065475, 0.12005, 0.072347],      
           [0.066838, 0.12459, 0.073908],      
           [0.068268, 0.1293, 0.075503],      
           [0.069792, 0.13418, 0.077268],      
           [0.071436, 0.13913, 0.079173],      
           [0.073206, 0.14428, 0.081195],      
           [0.075035, 0.14952, 0.083188],      
           [0.076994, 0.15483, 0.085288],      
           [0.079073, 0.16025, 0.087584],      
           [0.081282, 0.16579, 0.089893],      
           [0.083484, 0.17143, 0.09232],      
           [0.085728, 0.17716, 0.094882],      
           [0.088156, 0.18292, 0.097452],      
           [0.090584, 0.1888, 0.10012],      
           [0.093019, 0.19475, 0.1029],      
           [0.095617, 0.2007, 0.10572],      
           [0.098237, 0.20681, 0.10858],      
           [0.10086, 0.21289, 0.11157],      
           [0.1036, 0.21908, 0.11451],      
           [0.10636, 0.2253, 0.11758],      
           [0.10914, 0.2316, 0.12066],      
           [0.11197, 0.23791, 0.1238],      
           [0.11479, 0.24426, 0.12698],      
           [0.11773, 0.25069, 0.13025],      
           [0.12064, 0.25715, 0.13347],      
           [0.1236, 0.26366, 0.13674],      
           [0.12662, 0.27019, 0.14004],      
           [0.12966, 0.27675, 0.14341],      
           [0.13269, 0.28335, 0.14675],      
           [0.13572, 0.28998, 0.15017],      
           [0.13883, 0.29667, 0.15355],      
           [0.14195, 0.30336, 0.157],      
           [0.14509, 0.31012, 0.16044],      
           [0.1482, 0.31687, 0.16397],      
           [0.15136, 0.32367, 0.16745],      
           [0.15455, 0.33049, 0.17095],      
           [0.15775, 0.33736, 0.1745],      
           [0.16098, 0.34423, 0.17807],      
           [0.16424, 0.35115, 0.18159],      
           [0.16749, 0.35809, 0.18522],      
           [0.17077, 0.36506, 0.18882],      
           [0.17409, 0.37206, 0.19244],      
           [0.17746, 0.37909, 0.1961],      
           [0.18081, 0.38615, 0.19975],      
           [0.18425, 0.39325, 0.20347],      
           [0.18776, 0.4004, 0.20721],      
           [0.19125, 0.40756, 0.21096],      
           [0.19487, 0.41476, 0.21474],      
           [0.19852, 0.422, 0.21859],      
           [0.20228, 0.4293, 0.22246],      
           [0.20618, 0.43663, 0.22639],      
           [0.21013, 0.44402, 0.23034],      
           [0.21425, 0.45145, 0.23439],      
           [0.21851, 0.45893, 0.23847],      
           [0.22294, 0.46647, 0.24264],      
           [0.22756, 0.47407, 0.24689],      
           [0.23238, 0.48173, 0.25124],      
           [0.23745, 0.48944, 0.25565],      
           [0.24273, 0.49723, 0.26018],      
           [0.24834, 0.50507, 0.26479],      
           [0.25421, 0.51297, 0.26955],      
           [0.26039, 0.52093, 0.27439],      
           [0.26689, 0.52894, 0.27938],      
           [0.27376, 0.537, 0.28445],      
           [0.28094, 0.54509, 0.28968],      
           [0.28851, 0.55322, 0.29502],      
           [0.29643, 0.56137, 0.30049],      
           [0.30471, 0.56954, 0.3061],      
           [0.31334, 0.5777, 0.31178],      
           [0.32232, 0.58587, 0.31761],      
           [0.33161, 0.594, 0.32353],      
           [0.34124, 0.6021, 0.32956],      
           [0.35116, 0.61015, 0.33565],      
           [0.36134, 0.61814, 0.34183],      
           [0.37178, 0.62607, 0.34807],      
           [0.38245, 0.63392, 0.35437],      
           [0.39332, 0.64167, 0.3607],      
           [0.40437, 0.64934, 0.36709],      
           [0.41558, 0.6569, 0.37349],      
           [0.42692, 0.66435, 0.37991],      
           [0.43837, 0.67171, 0.38633],      
           [0.4499, 0.67896, 0.39277],      
           [0.46151, 0.68612, 0.39919],      
           [0.47317, 0.69317, 0.40562],      
           [0.48488, 0.70011, 0.41201],      
           [0.49662, 0.70698, 0.41839],      
           [0.50839, 0.71375, 0.42477],      
           [0.52015, 0.72044, 0.43112],      
           [0.53193, 0.72705, 0.43744],      
           [0.54371, 0.7336, 0.44374],      
           [0.55548, 0.74009, 0.45002],      
           [0.56726, 0.74651, 0.45628],      
           [0.57902, 0.75288, 0.46252],      
           [0.59077, 0.75922, 0.46875],      
           [0.60252, 0.76551, 0.47496],      
           [0.61425, 0.77177, 0.48113],      
           [0.62598, 0.778, 0.48732],      
           [0.63771, 0.7842, 0.49347],      
           [0.64942, 0.79038, 0.49962],      
           [0.66112, 0.79653, 0.50576],      
           [0.67283, 0.80268, 0.5119],      
           [0.68452, 0.80881, 0.51803],      
           [0.69622, 0.81492, 0.52415],      
           [0.7079, 0.82103, 0.53026],      
           [0.71959, 0.82713, 0.53638],      
           [0.73128, 0.83322, 0.54248],      
           [0.74297, 0.83931, 0.5486],      
           [0.75466, 0.84539, 0.5547],      
           [0.76635, 0.85147, 0.5608],      
           [0.77804, 0.85754, 0.56691],      
           [0.78974, 0.86362, 0.57302],      
           [0.80144, 0.86969, 0.57914],      
           [0.81315, 0.87576, 0.58525],      
           [0.82486, 0.88183, 0.59137],      
           [0.83658, 0.88789, 0.5975],      
           [0.8483, 0.89396, 0.60363],      
           [0.86003, 0.90003, 0.60976]]      
      
tofino_map = LinearSegmentedColormap.from_list('tofino', cm_data)      
# For use of "viscm view"      
test_cm = tofino_map      
      
if __name__ == "__main__":      
    import matplotlib.pyplot as plt      
    import numpy as np      
      
    try:      
        from viscm import viscm      
        viscm(tofino_map)      
    except ImportError:
        viscm = None
        print("viscm not found, falling back on simple display")      
        plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',      
                   cmap=tofino_map)      
    plt.show()      
